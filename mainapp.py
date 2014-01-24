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
#ROOT = '/Users/jasonbalkman/Documents/PYEC2/PROJECTS/basic_bs/'
ROOT = '/var/www/breast-density/'
DEBUG = True
FILE = 'static/img/IM-0001-3033.tif'

@app.route('/')
def hello_world():
   print 'Hello World!'
   return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_img():
   imgfile = request.args.get('imgfile')
   imgfilefull = os.path.join(ROOT, imgfile)
   print "Processing Image: "+imgfile
   imgfilefull_split = imgfile.rsplit('.')
   imgfilefull_new = imgfilefull_split[0]+'_seg.'+imgfilefull_split[1]
   imgfilefull_jpg = imgfilefull_split[0]+'_seg.jpg'

   origimg = cv2.imread(imgfilefull, cv2.CV_LOAD_IMAGE_GRAYSCALE)
   img = origimg[25:,:]
   imarray = np.array(img)
   imarraymarkup = imarray 
   maskarray = np.zeros_like(imarray)
   onesarray = np.ones_like(imarray)

   # Store dimensions for subsequent calculcations
   max_imheight = maskarray.shape[0]
   max_imwidth = maskarray.shape[1]
   
   if DEBUG: print max_imwidth, max_imheight
   
   #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   #ret,thresh = cv2.threshold(img,0,255,0)
   ret,thresh = cv2.threshold(imarray,0,255,0)

   contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

   biggest_contour = []
   for n, contour in enumerate(contours):
      if len(contour) > len(biggest_contour):
         biggest_contour = contour
         
   # Calculate R/L sidedness using centroid
   M = cv2.moments(biggest_contour)
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   right_side = cx > max_imwidth/2

   # Approximate the breast
   epsilon = 0.001*cv2.arcLength(biggest_contour,True)
   approx = cv2.approxPolyDP(biggest_contour,epsilon,True)

   # Calculate the hull and convexity defects
   drawhull = cv2.convexHull(approx)
   #cv2.drawContours(maskarray,drawhull,-1,(0,255,0),60)
   hull = cv2.convexHull(approx, returnPoints = False)
   defects = cv2.convexityDefects(approx,hull)
   
   # Plot the defects and find the most superior. Note: I think the superior and inferior ones have to be kept separate
   supdef_y = maskarray.shape[0]
   supdef_tuple = []
   
   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         if far[1] < cy and far[1] < supdef_y:
            supdef_y = far[1]
            supdef_tuple = far
            cv2.circle(maskarray,far,50,[0,0,255],-1)

   # Find lower defect if there is one
   infdef_y = 0
   infdef_tuple = []
   if defects is not None:
      for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         far = tuple(approx[f][0])
         if far[1] > infdef_y and supdef_tuple:
            if (right_side and far[0] > supdef_tuple[0]) or (not right_side and far[0] < supdef_tuple[0]):
               infdef_y = far[1]
               infdef_tuple = far

   # Try cropping contour beyond certain index; get indices of supdef/infdef tuples, and truncate vector beyond those indices
   cropped_contour = biggest_contour[:,:,:]

   if supdef_tuple:
      sup_idx = [i for i, v in enumerate(biggest_contour[:,0,:]) if v[0] == supdef_tuple[0] and v[1] == supdef_tuple[1]]
      if right_side:
         cropped_contour = cropped_contour[sup_idx[0]:,:,:]
      else:
         cropped_contour = cropped_contour[:sup_idx[0],:,:]
         
   if infdef_tuple:
      inf_idx = [i for i, v in enumerate(cropped_contour[:,0,:]) if v[0] == infdef_tuple[0] and v[1] == infdef_tuple[1]]
      if right_side:
         cropped_contour = cropped_contour[:inf_idx[0],:,:]
      else:
         cropped_contour = cropped_contour[inf_idx[0]:,:,:]
         
   if right_side:
      cropped_contour = cropped_contour[cropped_contour[:,0,1] != 1]
   else:
      cropped_contour = cropped_contour[cropped_contour[:,0,0] != 1]

   # Draw the cropped contour
   cv2.drawContours(imarraymarkup,cropped_contour,-1,(255,255,0),30)
   cv2.drawContours(imarraymarkup,biggest_contour,-1,(255,0,0),30)

   # Fill in the cropped polygon to mask
   #cv2.fillPoly(maskarray, pts = [cropped_contour], color=(255,255,255))
   cv2.fillPoly(maskarray, pts = [cropped_contour], color=(255,255,255))
   #maskarray = ~np.all(maskarray == 0, axis=1)a
   #print maskarray
   #maskarray[~np.all(maskarray == 0, axis=2)]

   # Threshold the cropped version
   #ret_otsu,thresh_otsu = cv2.threshold(imarray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

   # Multiple original image to the mask to get the cropped image
   imarray2 = imarray + onesarray
   print imarray2
   imcrop_orig = cv2.bitwise_and(imarray2, maskarray)
   #cv2.drawContours(maskarray,biggest_contour,-1,(255,0,0),3)

   output1 = rank.minimum(imcrop_orig, disk(20))

   thresh = mahotas.otsu(output1, ignore_zeros = True)
   output2 = output1 > thresh

   # Draw thick black contour to eliminate the skin and nipple from the image
   #cv2.drawContours(imcrop_orig,cropped_contour,-1,(0,0,0),200)
   
   # Create pil images
   #pilimg = Image.fromarray(output2)
   #pilimg.save(imgfilefull_new)
   #pilimg.save(imgfilefull_jpg)
   
   #pil_imcrop_orig = Image.fromarray(imcrop_orig)
   #pil_orig = Image.fromarray(img)
   #pil_imcrop_min = Image.fromarray(output1)
   pil_markup = Image.fromarray(maskarray)

   plt.figure(figsize=(20,10))
   plt.subplot(1,3,1),plt.imshow(img,'gray')
   plt.title('Original')
   plt.subplot(1,3,2),plt.imshow(pil_markup)
   plt.title('Breast Segmentation')
   plt.subplot(1,3,3),plt.imshow(output2,'gray')
   plt.title('Binary Breast Density')
   plt.savefig(imgfilefull_new)
   plt.savefig('uploads/output.jpg')

   imgfilefullsplit_jpg = imgfilefull_jpg.split('/')

   #responsefile = imgfilefullsplit_jpg[-2]+'/'+imgfilefullsplit_jpg[-1]
   responsefile = ROOT+'uploads/output.jpg'

   print "Response File: "+responsefile
   return jsonify({"success":True, "file": responsefile})

@app.route('/serve_img')
def serve_img():
   fname = request.args.get('file')
   full_path = os.path.join(ROOT, fname)
   print full_path
   return send_file(full_path, mimetype='image/jpeg')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            now = datetime.now()
            ext = file.filename.rsplit('.', 1)[1]
            filename_noext = os.path.join(app.config['UPLOAD_FOLDER'], "%s" % (now.strftime("%Y-%m-%d-%H-%M-%S-%f")))
            filename_ext = filename_noext+'.'+ext
            file.save(filename_ext)
            if istiff(file.filename):
               im = Image.open(filename_ext)
               filename_jpg = filename_noext+'.jpg'
               im.save(filename_jpg) 
            else:
               filename_jpg = filename_ext

            return jsonify({"success":True, "file": filename_jpg})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def istiff(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in TIFF_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)

# Unused Functions
def serve_pil_image(pil_img):
    img_io = StringIO.StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    print "return send file..."
    return send_file(img_io, mimetype='image/jpeg')
