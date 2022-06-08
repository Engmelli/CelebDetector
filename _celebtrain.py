import os
import cv2 as cv
import numpy as np

celebs = ['Christian Bale', 'Guido van Rossum', 'Playboi Carti', 'Robert Pattinson', 'Volodymyr Zelenskyy']

dir = r'D:\Downloads\Celebs'

    haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces = []
labels = []

def train():
    for person in celebs:
        path = os.path.join(dir, person)
        label = celebs.index(person)

        for img in os.listdir(path):

            imagepath = os.path.join(path, img)

            img = cv.imread(imagepath)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            faces_detect = haar_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces_detect:
                face = gray[y:y+h, x:x+h]
                faces.append(face)
                labels.append(label)

train()

faces = np.array(faces, dtype= 'object')
labels = np.array(labels)

face_recon = cv.face.LBPHFaceRecognizer_create()
face_recon.train(faces, labels)
face_recon.save('celebrecon.yml')

np.save('celebfaces', faces)
np.save('celebnames', labels)





