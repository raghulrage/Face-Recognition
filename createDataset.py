import cv2

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Id = int(input('Enter an Id: '))

itrId = 0

while True:
    
    ret,image = camera.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    face = detector.detectMultiScale( gray, scaleFactor = 1.3, minNeighbors = 5)

    itrId += 1

    for (x,y,w,h) in face:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imwrite('dataset/User.'+str(Id)+'.'+str(itrId)+'.jpg',gray[y:y+h,x:x+w])

        cv2.imshow('Video',image)

    if cv2.waitKey(100) and 0xFF == 27:
        break

    if itrId > 20:
        break

camera.release()
cv2.destroyAllWindows()

