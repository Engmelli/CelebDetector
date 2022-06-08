import cv2 as cv
import numpy as np

celebs = ['Christian Bale', 'Guido van Rossum', 'Playboi Carti', 'Robert Pattinson', 'Volodymyr Zelenskyy']

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces = np.load('celebfaces.npy', allow_pickle= True)
labels = np.load('celebnames.npy')

celebrecon = cv.face.LBPHFaceRecognizer_create()
celebrecon.read('celebrecon.yml')

webcamfootage = cv.VideoCapture(0)

while True:

    isTrue, frame = webcamfootage.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_detect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_detect:
        face = gray[y:y+h, x:x+h]

        label, confidence = celebrecon.predict(face)

        nameandconfidence = str(celebs[label]) + fr' with {int(confidence)}% confidence'

        cv.putText(frame, nameandconfidence, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0))
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness= 2)

    cv.imshow("Celebrity recognizer", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break


video.release()
cv.destroyAllWindows()


