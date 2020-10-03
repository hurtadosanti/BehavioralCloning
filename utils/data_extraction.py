from zipfile import ZipFile
import csv
import cv2
import pickle
import numpy as np

with ZipFile('data.zip', 'r') as zipObj:
   zipObj.extractall('./')
   print('Finish extracting')

images = []
measurements = []
with open('./data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    header = True
    for l in reader:
        if header:
            header = False
        else:
            path = './data/'+l[0]
            image = cv2.imread(path)
            if image is not None :
                # normal image
                images.append(image)
                measure = float(l[3])
                measurements.append(measure)
            else:
                print('Image not read',path)

# Processing in place for better memory management
assert len(images)==len(measurements)
print('images read:',len(images))
X_train =np.array(images)
y_train = np.array(measurements)
del images
del measurements
assert len(X_train)==len(y_train)
X_train = np.sum(X_train/3,axis=3,keepdims=True)
X_train = (X_train-128)/128

pickle.dump(X_train, open("./data/x_train.b", "wb"))
pickle.dump(y_train, open("./data/y_train.b", "wb"))
print('data persisted')