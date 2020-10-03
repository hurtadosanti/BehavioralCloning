import cv2
import numpy as np
import pandas as pd

def image_generator(logs, batch_size, mode="train"):
    while True:
        start = 0
        end = batch_size
        while start  < len(logs):
            selected = logs[start:end]
            images,labels = batch_loader(selected)
            #x = images[start:end]
            #y = labels[start:end]
            yield images,labels
            start += batch_size
            end += batch_size

def batch_loader(log_select):
    images = []
    measurements = []
    for l in log_select:
        path = './data/'+l['center']
        image = cv2.imread(path)
        if image is not None :
            # normal image
            images.append(image)
            measure = l['steering']
            measurements.append(measure)
        else:
            print('Image not read',path)
    return np.asarray(images),np.asarray(measurements)