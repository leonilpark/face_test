import cv2
import numpy as np
import os, glob


class module_face_detection:

    def __init__(self,base_dir):
        self.train_data, self.train_labels = [], []
        self.dirs = [d for d in glob.glob(base_dir+'/*') if os.path.isdir(d)]

    def process(self):
        print("Collecting train data set:")
        for dir in self.dirs:
            id = dir.split('_')[1]  # Excepting id
            files = glob.glob(dir + '/*.jpg')
            print('\t path:%s, %dfiles' % (dir, len(files)))
            for file in files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                # image -> train_data, id -> train_labels
                self.train_data.append(np.asarray(img, dtype=np.uint8))
                self.train_labels.append(int(id))

            # Convert Numpy array
            train_data = np.asarray(self.train_data)
            train_labels = np.int32(self.train_labels)

            # Creating LBP face detection & Training
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(train_data, train_labels)
            model.write('model/custom/custom.xml')