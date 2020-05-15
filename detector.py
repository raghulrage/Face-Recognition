import cv2

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = detector.detectMultiScale(gray,scaleFactor = 1.3, minNeighbors = 5, minSize = (30,30))

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

        faceid, percentage = recognizer.predict(gray[y:y+h,x:x+w])

        if percentage > 50:
            text = 'Unknown'
        else:
            text = str(faceid)

        cv2.putText(img, text, (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Video',img)

    if cv2.waitKey(100) & 0xFF == 27:
        break

camera.release()
cv2.destroyAlLWindows()


