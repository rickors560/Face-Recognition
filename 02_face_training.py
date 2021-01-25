import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'
# Local Binary Patterns Histograms 
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    #imagePaths = [os.path.join(path,f) for f in os.listdir(path)]   
    parent_path = [] 
    for f in os.listdir(path):
        parent_path.append(path+'/'+f)
    imagePaths = []
    for j in range(len(parent_path)):
        for i in os.listdir(parent_path[j]):
            imagePaths.append(str(parent_path[j])+'/'+str(i))
    faceSamples=[]
    ids = []
    #print(imagePaths)
    for imagePath in imagePaths:

        img_numpy = cv2.imread(imagePath, 0)# convert it to grayscale

        id = int(imagePath.split('/')[1].split('.')[1])
        
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
