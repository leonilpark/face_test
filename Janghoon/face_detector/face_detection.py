import cv2
import numpy as np
import os, glob

# setting
base_dir = 'dataset'
train_data, train_labels = [], []
dirs = [d for d in glob.glob(base_dir+'/*') if os.path.isdir(d)]

print("Collecting train data set:")
for dir in dirs:
    id = dir.split('_')[1] # Excepting id
    files = glob.glob(dir+'/*.jpg')
    print('\t path:%s, %dfiles' % (dir, len(files)))
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # image -> train_data, id -> train_labels
        train_data.append(np.asarray(img, dtype=np.uint8))
        train_labels.append(int(id))

    # Convert Numpy array
    train_data = np.asarray(train_data)
    train_labels = np.int32(train_labels)

    # Creating LBP face detection & Training
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(train_data, train_labels)
    model.write('model/custom/custom.xml')
    

