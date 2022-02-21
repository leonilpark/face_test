import cv2
import os


# setting
base_dir = 'dataset'  # saving photos
target_cnt = 200 # number of photos
cnt = 0 # number of collected photos

# creating descriptor
face_classifier = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

# input user name and number to make directory
name = input("Insert User Name : ")
id = input("Insert User Id : ")
dir = os.path.join(base_dir, name+'_'+id)
if not os.path.exists(dir):
    os.mkdir(dir)

# Camera Capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # face detection
        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.2,  # 검색 윈도우 확대 비율
            minNeighbors=6,  # 최소 간격
            minSize=(20, 20)  # 최소 크기
        )
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            # marking area and save file
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            file_name_path = os.path.join(dir, str(cnt) + '.jpg')
            cv2.imwrite(file_name_path, face)
            cv2.putText(frame, str(cnt), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cnt += 1
        else:
            # if faces are more than 1 or None Detection, show error
            if len(faces) == 0:
                msg = "no face"
            else:
                msg = "too many face."
            cv2.putText(frame, str(msg), (445, 430), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('face record', frame)
        if cv2.waitKey(1) == 27 or cnt == target_cnt:
            break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Completed")
