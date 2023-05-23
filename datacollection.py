import cv2
import os
cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
face_id = input('enter user id end press <return> ==>  ')
print("Initializing face capture. Look the camera and wait ...")
count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1) # 1 stright, 0 reverse
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        cv2.imwrite("datasets/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 100: # Take 100 face sample and stop video
         break
cam.release()
cv2.destroyAllWindows()