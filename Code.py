#Code Author Shounak Chandra
#Brain tumor Detection project with Python and Deep machine learning and AI 
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
path = filedialog.askopenfilename()
 
def select_image():
  # grab a reference to the image panels
  global panelA, panelB
 
  # open a file chooser dialog and allow the user to select an input
  # image
  path = filedialog.askopenfilename()
    # ensure a file path was selected
  if len(path) > 0:
    # load the image from disk, convert it to grayscale, and detect
    # edges in it
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 100)
 
    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    # convert the images to PIL format...
    image = Image.fromarray(image)
    edged = Image.fromarray(edged)
 
    # ...and then to ImageTk format
    image = ImageTk.PhotoImage(image)
    edged = ImageTk.PhotoImage(edged)
        # if the panels are None, initialize them
    if panelA is None or panelB is None:
      # the first panel will store our original image
      panelA = Label(image=image)
      panelA.image = image
      panelA.pack(side="left", padx=10, pady=10)
 
      # while the second panel will store the edge map
      panelB = Label(image=edged)
      panelB.image = edged
      panelB.pack(side="right", padx=10, pady=10)
 
    # otherwise, update the image panels
    else:
      # update the pannels
      panelA.configure(image=image)
      panelB.configure(image=edged)
      panelA.image = image
      panelB.image = edged
      # initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
 
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
# kick off the GUI
root.mainloop()

def ShowImage(title,img,ctype):

  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       
    rgb_img = cv2.merge([r,g,b])     
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()
img           = cv2.imread(path)
gray          = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ShowImage('Brain MRI',gray,'gray')

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
ShowImage('Thresholding image',thresh,'gray')

ret, markers = cv2.connectedComponents(thresh)


marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 

largest_component = np.argmax(marker_area)+1 
brain_mask = markers==largest_component

brain_out = img.copy()

brain_out[brain_mask==False] = (0,0,0)

img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    

sure_bg = cv2.dilate(opening,kernel,iterations=3)
    

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
   

ret, markers = cv2.connectedComponents(sure_fg)
     

markers = markers+1

     

markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

im1 = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
ShowImage('Watershed segmented image',im1,'gray')

brain_mask = np.uint8(brain_mask)
kernel = np.ones((8,8),np.uint8)
closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
ShowImage('Closing', closing, 'gray')

brain_out = img.copy()

brain_out[closing==False] = (0,0,0)


