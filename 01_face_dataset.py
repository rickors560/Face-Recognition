import cv2
import pandas as pd
import os
import csv

cam = cv2.VideoCapture(0)


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
dataset = pd.read_csv('DATA.csv')
data_id = dataset.iloc[:, 0:-1].values
d_id = str(int(data_id[-1])+1)
face_id = d_id
name = input('Enter your name ::').upper()
with open('DATA.csv', 'a+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([d_id, name])
directory = "User."+face_id+'.'+name
parent_dir = "/media/alfa/DATA/Project/FacialRecognition/dataset"
path = os.path.join(parent_dir, directory) 
os.mkdir(path)
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/"+directory+"/" + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100)# Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


