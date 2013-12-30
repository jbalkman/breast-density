# General
import os
import StringIO
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request, send_file

# Image Processing
import numpy
import matplotlib.pyplot as plt

from PIL import Image
from skimage import data, io, filter
from skimage.morphology import watershed
from skimage import exposure

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff']
TIFF_EXTENSIONS = ['tif', 'tiff']

@app.route('/')
def hello_world():
   print 'Hello World!'
   return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_img():
   imgfile = request.args.get('imgfile')
   imgfilefull = os.path.join('/Users/jasonbalkman/Documents/PYEC2/PROJECTS/basic/', imgfile)
   print "Processing Image: "+imgfile
   imgfilefull_split = imgfile.rsplit('.')
   imgfilefull_new = imgfilefull_split[0]+'_seg.'+imgfilefull_split[1]
   imgfilefull_jpg = imgfilefull_split[0]+'_seg.jpg'
   im = Image.open(imgfilefull)
   imarray = numpy.array(im)
   elevation_map = filter.sobel(imarray)
   markers = numpy.zeros_like(imarray)
   markers[imarray < 5] = 1
   markers[imarray > 10] = 255
   print "Running Watershed segmentation..."
   segmentation = watershed(elevation_map, markers)
   #segmented_rs = exposure.rescale_intensity(segmentation, in_range=(0, 255))
   #plt.imshow(segmented_rs)
   #io.imshow(segmentation)
   print "Converting image back to PIL and saving..."
   pilimg = Image.fromarray(segmentation)
   pilimg.save(imgfilefull_new)
   pilimg.save(imgfilefull_jpg)
   imgfilefullsplit_jpg = imgfilefull_jpg.split('/')
   responsefile = imgfilefullsplit_jpg[-2]+'/'+imgfilefullsplit_jpg[-1]
   print "Response File: "+responsefile
   return jsonify({"success":True, "file": responsefile})
   
@app.route('/serve_img')
def serve_img():
   fname = request.args.get('file')
   full_path = os.path.join('/Users/jasonbalkman/Documents/PYEC2/PROJECTS/basic/', fname)
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

