import cv2
from random import randrange as r
#dataset load
trainedData = cv2.CascadeClassifier('Face.xml')
#choose a image
img = cv2.imread('sample image.jpg')
#conversion to black n white (grayscale)
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#detect faces
faceCoordinates = trainedData.detectMultiScale(grayimg)
for x,y,w,h in faceCoordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (r(0,256), r(0,256), r(0,256)), 2)
#display image
cv2.imshow('Single Person', img)
#pause execution of program until any key is pressed
cv2.waitKey()
print("End of Program")


