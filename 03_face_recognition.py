import pandas as pd
import cv2
import numpy as np
import os 
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0
img_name = '--unknown--'
dataset = pd.read_csv('DATA.csv')
id_data = dataset.iloc[:, :-1].values
names_data = dataset.iloc[:, -1].values
ids = [str(id_data[i][j]) for i in range(len(id_data)) for j in range(len(id_data[i]))]
names = [str(names_data[i]) for i in range(len(names_data))]

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)


t_end = time.time() + 10
while time.time() < t_end:

    ret, img = cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            img_name = names[id-1]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            img_name = "--unknown--"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(img_name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
        cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
print(img_name)
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
