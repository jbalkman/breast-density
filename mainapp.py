# General
import os
import StringIO
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request, send_file

# Image Processing
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mahotas

from PIL import Image
from scipy import ndimage
from skimage.morphology import watershed, disk
from skimage import data
from skimage.filter import rank, threshold_otsu
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff', 'dcm']
TIFF_EXTENSIONS = ['tif', 'tiff']
DEBUG = True
FILE = 'static/img/IM-0001-3033.tif'

# These may not be needed if we're dealing with relative files...seems ok to leave out
#ROOT = '/Users/jasonbalkman/Documents/PYEC2/PROJECTS/basic_bs/'
#ROOT = '/var/www/breast-density/'

@app.route('/')
def hello_world():
   print 'Hello World!'
   return render_template('index.html')

@app.route('/process_serve', methods=['GET'])
def process_serve_img():
   imgfile = request.args.get('imgfile')
   print "Process/Serving Image: "+imgfile
   imgprefix = imgfile.rsplit('.')[0]

   d, c, s, v = processFile(imgprefix) # returns density, density category, side, and view

   with open(imgprefix+"-out.jpg", "rb") as f: # the imgfile has been resaved as the results from the processing above, so there is no need to change the file
      data = f.read()
      print "Removing full path: "+imgfile
      os.remove(imgprefix+"-out.jpg")

   # Clean-up upload files so nothing is left on the server
   try:
      print "Removing Prefix Files: "+imgprefix
      os.remove(imgprefix+'.jpg')
      os.remove(imgprefix+'.tif')
   except:
      print "Unable to remove file "+imgprefix
      
   return jsonify({"success":True, "imagefile": data.encode("base64"), "density":d, "dcat":c, "side":s, "view":v})

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            now = datetime.now()
            ext = file.filename.rsplit('.', 1)[1]
            filename_noext = os.path.join(app.config['UPLOAD_FOLDER'], "%s" % (now.strftime("%Y-%m-%d-%H-%M-%S-%f")))
            filename_ext = filename_noext+'.'+ext
            file.save(filename_ext) # saving the original filename ?needed
            if istiff(file.filename):
               im = Image.open(filename_ext)
               filename_jpg = filename_noext+'.jpg'
               im.save(filename_jpg) # saving the jpg file ?needed
            else:
               filename_jpg = filename_ext

            return jsonify({"success":True, "file": filename_jpg})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def istiff(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in TIFF_EXTENSIONS

def processFile(f):
   
   fname = f+".jpg"
   fname_out = f+"-out.jpg"
   
   print "File to process: "+fname
   origimg = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
   
   # Chop off the top of the image b/c there is often noncontributory artifact & make numpy arrays
   img = origimg[25:,:]
   imarray = np.array(img)
   
   imarraymarkup = imarray
   maskarray = np.zeros_like(imarray)
   contoursarray = np.zeros_like(imarray)
   onesarray = np.ones_like(imarray)
   
    # Store dimensions for subsequent calculcations
   max_imheight = maskarray.shape[0]
   max_imwidth = maskarray.shape[1]
   
   if DEBUG: print max_imwidth, max_imheight
    
   # Choose the minimum in the entire array as the threshold value b/c some mammograms have > 0 background which screws up the contour finding if based on zero or some arbitrary number
   ret,thresh = cv2.threshold(imarray,np.amin(imarray),255,cv2.THRESH_BINARY)
   contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   biggest_contour = []
   for n, contour in enumerate(contours):
      if len(contour) > len(biggest_contour):
         biggest_contour = contour

    # Get the lower most extent of the contour (biggest y-value)
   max_vals = np.argmax(biggest_contour, axis = 0)
   min_vals = np.argmin(biggest_contour, axis = 0)
    #print max_vals[0,1]
   bc_max_y = biggest_contour[max_vals[0,1],0,1] # get the biggest contour max y
   bc_min_y = biggest_contour[min_vals[0,1],0,1] # get the biggest contour min y
    #print "Biggest Contour Max Y:"
    #print bc_max_y
    #print "Biggest Contour Min Y:"
    #print bc_min_y
   
   cv2.drawContours(contoursarray,biggest_contour,-1,(255,255,255),15)            

    # Calculate R/L sidedness using centroid
   M = cv2.moments(biggest_contour)
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   right_side = cx > max_imwidth/2
    
    # Plot the center of mass
   cv2.circle(contoursarray,(cx,cy),100,[255,0,255],-1)            

    # Approximate the breast
   epsilon = 0.001*cv2.arcLength(biggest_contour,True)
   approx = cv2.approxPolyDP(biggest_contour,epsilon,True)
            
    # Calculate the hull and convexity defects
   drawhull = cv2.convexHull(approx)
    #cv2.drawContours(contoursarray,drawhull,-1,(0,255,0),60)
   hull = cv2.convexHull(approx, returnPoints = False)
   defects = cv2.convexityDefects(approx,hull)
   
    # Plot the defects and find the most superior. Note: I think the superior and inferior ones have to be kept separate
    # Also make sure that these are one beyond a reasonable distance from the centroid (arbitrarily cdist_factor = 80%) to make sure that nipple-related defects don't interfere
   supdef_y = maskarray.shape[0]
   supdef_tuple = []
   
   cdist_factor = 0.80

   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         if far[1] < (cy*cdist_factor) and far[1] < supdef_y:
            supdef_y = far[1]
            supdef_tuple = far
            cv2.circle(contoursarray,far,50,[255,0,255],-1)

    # Find lower defect if there is one
    # Considering adding if a lower one is at least greater than 1/2 the distance between the centroid and the lower most border of the contour (see IMGS_MLO/IM4010.tif)
   infdef_y = 0
   infdef_tuple = []
   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         if far[1] > infdef_y and supdef_tuple: # cy + 3/4*(bc_max_y - cy) = (bc_max_y + cy)/2
            if (right_side and far[0] > supdef_tuple[0]) or (not right_side and far[0] < supdef_tuple[0]):
               infdef_y = far[1]
               infdef_tuple = far
               cv2.circle(contoursarray,far,50,[255,0,255],-1)

    # Try cropping contour beyond certain index; get indices of supdef/infdef tuples, and truncate vector beyond those indices
   cropped_contour = biggest_contour[:,:,:]
               
   if supdef_tuple:
      sup_idx = [i for i, v in enumerate(biggest_contour[:,0,:]) if v[0] == supdef_tuple[0] and v[1] == supdef_tuple[1]]
      if sup_idx:
         if right_side:
            cropped_contour = cropped_contour[sup_idx[0]:,:,:]
         else:
            cropped_contour = cropped_contour[:sup_idx[0],:,:]
            
   if infdef_tuple:
      inf_idx = [i for i, v in enumerate(cropped_contour[:,0,:]) if v[0] == infdef_tuple[0] and v[1] == infdef_tuple[1]]
      if inf_idx:
         if right_side:
            cropped_contour = cropped_contour[:inf_idx[0],:,:]
         else:
            cropped_contour = cropped_contour[inf_idx[0]:,:,:]
         
   if right_side:
      cropped_contour = cropped_contour[cropped_contour[:,0,1] != 1]
   else:
      cropped_contour = cropped_contour[cropped_contour[:,0,0] != 1]

    # Draw the cropped contour
    #cv2.drawContours(imarraymarkup,cropped_contour,-1,(255,255,0),30)
    #cv2.drawContours(imarraymarkup,biggest_contour,-1,(255,0,0),30)

    # Fill in the cropped polygon to mask
    #cv2.fillPoly(maskarray, pts = [cropped_contour], color=(255,255,255))
   cv2.fillPoly(maskarray, pts = [cropped_contour], color=(255,255,255))
    #maskarray = ~np.all(maskarray == 0, axis=1)a
    #print maskarray
    #maskarray[~np.all(maskarray == 0, axis=2)]

    # Threshold the cropped version
    #ret_otsu,thresh_otsu = cv2.threshold(imarray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Multiply original image to the mask to get the cropped image
   imarray2 = imarray + onesarray
   imcrop_orig = cv2.bitwise_and(imarray2, maskarray)
    #cv2.drawContours(maskarray,biggest_contour,-1,(255,0,0),3)
    
   rankmin_out = rank.minimum(imcrop_orig, disk(20))
   thresh = mahotas.otsu(rankmin_out, ignore_zeros = True)

    # Draw thick black contour to eliminate the skin and nipple from the image
   cv2.drawContours(rankmin_out,cropped_contour,-1,(0,0,0),255) # 
    #cv2.drawContours(maskarray,cropped_contour,-1,(0,0,0),255) # 

    # Apply the thresholding to generate a new matrix and convert to int type
   otsubool_out = rankmin_out > thresh
   otsubinary_out = otsubool_out.astype('uint8')
   otsuint_out = otsubinary_out * 255

    # Crop out the fibroglandular tissue
    #print output2.shape, imarray2.shape, np.amax(output2), np.amax(imarray2), np.amax(maskarray)
   imcrop_fgt = cv2.bitwise_and(imarray2, otsuint_out) # both arrays are uint8 type

   segmented = maskarray > 0
   segmented = segmented.astype(int)
   segmented_sum = segmented.sum()
   otsubinary_sum = otsubinary_out.sum()
   
   density = (otsubinary_sum*100/segmented_sum).astype(int)
   
   if density < 25:
      dcat = 'Fatty'
   elif density < 50:
      dcat = 'Scattered'
   elif density < 75:
      dcat = 'Heterogenous'
   else:
      dcat = 'Extremely Dense'
        
   if right_side:
      side = 'Right'
   else:
      side = 'Left'

   if bc_min_y > 1:
      view = 'CC'
   else:
      view = 'MLO'
      
   avg = (imcrop_fgt.sum()/otsubinary_sum).astype(int)
   print side, view, otsubinary_sum, segmented_sum, density, dcat, avg, avg/np.amax(imcrop_fgt), np.amax(imcrop_fgt)
    
    # Create pil images
   #pilimg = Image.fromarray(output2)
   #pilimg.save(imgfilefull_new)
   #pilimg.save(imgfilefull_jpg)
    
   #pil_imcrop_orig = Image.fromarray(imcrop_orig)
   #pil_orig = Image.fromarray(img)
   #pil_imcrop_min = Image.fromarray(output1)

    # Plot a 4x4
   pil_imarray2 = Image.fromarray(imarray2)
   pil_segmented = Image.fromarray(maskarray)
   pil_markup = Image.fromarray(contoursarray)
   pil_fgt = Image.fromarray(imcrop_fgt)
   
   plt.figure(figsize=(20,10))
   plt.subplot(1,4,1),plt.imshow(pil_imarray2,'gray')
   plt.title('Original')
   plt.subplot(1,4,2),plt.imshow(pil_segmented, 'gray')
   plt.title('Breast Segmentation')
   plt.subplot(1,4,3),plt.imshow(otsuint_out, 'gray')
   plt.title('Fibroglandular Segmentation')
   plt.subplot(1,4,4),plt.imshow(imcrop_fgt,'gray')
   plt.title('Fibroglandular Tissue')
    #plt.show()

    # Plot a 2x2
   '''pil_segmented = Image.fromarray(maskarray)
   pil_markup = Image.fromarray(contoursarray)
   
   plt.figure(figsize=(20,20))
   plt.subplot(1,2,1),plt.imshow(img,'gray')
   plt.title('Original')
   plt.subplot(1,2,2),plt.imshow(pil_markup)
   plt.title('Contours')
   plt.show()'''

   plt.savefig(fname_out)
    
   return density, dcat, side, view

if __name__ == '__main__':
    app.run(debug=True)

# Unused Functions
def serve_pil_image(pil_img):
    img_io = StringIO.StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    print "return send file..."
    return send_file(img_io, mimetype='image/jpeg')
