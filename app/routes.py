from flask import Flask,render_template,url_for,request
import os
from app import app
import cv2
import numpy as np

@app.route('/')
@app.route('/options',methods=['GET','POST'])
def options():
    return render_template('options.html')

UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/scaling',methods=['GET','POST'])
def scaling():
    
    #f=request.files['file']
    #f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    #img = cv2.imread(full_filename)
    #res = cv2.resize(img,(2*width, 2*height))
    return render_template('scaling.html')

@app.route('/rotation',methods=['GET','POST'])
def rotation():
    if request.method == "POST":
      f=request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
      full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
      img = cv2.imread(full_filename)
      rows,cols = img.shape
      M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
      dst = cv2.warpAffine(img,M,(cols,rows))
      return render_template('rotation.html')

@app.route('/grayconversion',methods=['GET','POST'])
def grayconversion():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    render_template('grayconverison.html')

@app.route('/facedetection',methods=['GET','POST'])
def facedetection():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
       res_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = img[y:y+h, x:x+w]
       eyes = eye_cascade.detectMultiScale(roi_gray)
       for (ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)    
    render_template('facedetction.html')

@app.route('/edgedetection',methods=['GET','POST'])           
def edgedetection():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename) 
    edges = cv2.Canny(img,100,200)
    render_template('edgedetection.html')

@app.route('/cornerdetection',methods=['GET','POST'])
def cornerdetection():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    render_template('cornerdetection.html')


@app.route('/linedetection',methods=['GET','POST'])
def linedetection():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
       a = np.cos(theta)
       b = np.sin(theta)
       x0 = a*rho
       y0 = b*rho
       x1 = int(x0 + 1000*(-b))
       y1 = int(y0 + 1000*(a))
       x2 = int(x0 - 1000*(-b))
       y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)     
    cv2.imwrite('linedetection.jpg',img)
    render_template('linedetection.html')     

@app.route('/circledetection',methods=['GET','POST'])
def circledetection():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)   
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    render_template('circledetection.html')

@app.route('/imagesegmentation',methods=['GET','POST'])
def imagesegmentation():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)   
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

# Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    render_template('imagesegmentation.html')

@app.route('/erosion',methods=['GET','POST'])
def erosion():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)   
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    render_template('erosion.html')

@app.route('/dilation',methods=['GET','POST'])
def dilation():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)  
    kernel = np.ones((5,5),np.float32)/25
    dilate = cv2.dilate(img,kernel,iterations = 1)
    render_template('dilation.html')

@app.route('/blurring',methods=['GET','POST'])
def blurring():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)   
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    render_template('blurring.html')

@app.route('/foregroundextraction',methods=['GET','POST'])
def foregroundextraction():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    res = img*mask2[:,:,np.newaxis]   
    render_template('foregroundextraction')

@app.route('/laplacianderivative',methods=['GET','POST'])
def laplacianderivative():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)  
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    render_template('laplacianderivative.html')


@app.route('/sobelderivative',methods=['GET','POST'])
def sobelderivative():
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)   
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    render_template('sobelderivative.html')

@app.route('/masking',methods=['GET','POST'])
def masking(): 
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img = cv2.imread(full_filename)                                  

if __name__== "__main__":
    app.run(debug=True)