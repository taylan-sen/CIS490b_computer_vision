# facedetecthaar.py
#
# This program uses opencv's haar cascade face detector to find bounding boxes
# of all faces in the provided image.
#
# Code is modified from opencv example at:
#   https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html 

import cv2

# make sure the xml model file is in same directory as this program.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('monalisa.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray)

# Detect faces
#   detectMultScale returns an 2d ndarray
faces = face_cascade.detectMultiScale(gray)
print('detected face(s) at:', faces)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 5)
  cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 0), 5)

# Display the output
cv2.imshow('press any key to exit', img)
cv2.imwrite('monalisa_out.jpg',img)
cv2.waitKey(0) # blocks until a key is pressed when window is active.
cv2.destroyAllWindows()
