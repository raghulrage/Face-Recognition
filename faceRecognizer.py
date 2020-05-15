import cv2
import os
from PIL import Image
import numpy as np

#initialization
faceArr, ids = [], []

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#get path of all images
imgPaths = [os.path.join('dataset',f) for f in os.listdir('dataset')]

for imgPath in imgPaths:

    #if not image file continue
    if os.path.split(imgPath)[-1].split('.')[-1] != 'jpg' :
        continue

    #convert to grayscale
    img = Image.open(imgPath).convert('L')

    #img to array
    imgArr = np.array(img, 'uint8')

    faceId = int(os.path.split(imgPath)[-1].split('.')[1])

    #extract face using cascade classifier
    face = detector.detectMultiScale(imgArr)

    for (x,y,w,h) in face:
        faceArr.append(imgArr[y:y+h,x:x+w])
        ids.append(faceId)


recognizer.train(faceArr, np.array(ids))
recognizer.save('trainner/trainner.yml')
