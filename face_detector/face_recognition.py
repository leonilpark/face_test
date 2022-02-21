import cv2
import numpy as np
import os, glob


# setting
base_dir = 'model/custom'
min_accuracy = 80 # threshold
face_classifier = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read(os.path.join(base_dir, 'custom.xml'))

# information creating
dirs = [ d for d in glob.glob(base_dir + "/*") if os.path.isdir(d)]
names = dict([])
for dir in dirs:
    dir = os.path.basename(dir)
    name, id = dir.split('_')
    names[int(id)] = name

# capture setting on camera
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("no frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detecting
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # draw face area
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.resize(face, cv2.COLOR_BGR2GRAY)
        # prediction with recognition
        label, confidence = model.predict(face)
        if confidence < 400:
            # accuracy distance -> procent
            accuracy = int( 100 * (1-confidence/400))
            if accuracy >= min_accuracy:
                msg = '%s(%.0f%%)'%(names[label], accuracy)
                # display name and accuracy
                txt, base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 1, 3)
                cv2.rectangle(frame, (x, y-base-txt[1]), (x+txt[0], y+txt[1]),(0, 255, 255), -1)
                cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 2, cv2.LINE_AA)
            else: # mosaic
                rate = 15 # reduction ratio in mosaic
                x = x-5
                y = y - 15
                w = w + 10
                h = h + 30
                roi = face
                roi = cv2.resize(roi, (w//rate, h//rate))
                # enlarge to original size
                roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
                frame[y:y+h, x:x+w] = roi # mosaic application
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == 27: # esc
        break
    cap.release()
    cv2.destroyAllWindows()


