from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os
import cv2
import matlab.engine
import numpy as np
import re
import sys
import getopt
from keras.models import load_model


eng = matlab.engine.start_matlab()
eng.addpath(r'/home/rpandey/people_detect/edge_boxes/edges',nargout=0)
model  = load_model('/data/stars/share/people_depth/people-depth/fulldata/depth_people_alexnet.h5')
model.load_weights("/home/rpandey/people_detect/weights/weights-improvement-20-0.50.hdf5")
ebox_min_threshold = 0.6

    
def TestData(filepath, img_type="depth"):
    
    for images in os.listdir(filepath):
        if images.endswith(img_type+".jpg"):
            imgpath = os.path.join(filepath, images)
            img_raw = cv2.imread(imgpath, 2)
            img = img_raw.astype(np.float32)
            img -= np.min(img)
            img /= (np.max(img) - np.min(img))
            img *= 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bboxes = eng.getEdgeBoxes(imgpath)
            max_val = -1
            min_val = 2
            bboxes_use = []
            result_X = []
            for bbox in bboxes:
                if bbox[4] < min_val:
                    min_val = bbox[4]
                if bbox[4] > max_val:
                    max_val = bbox[4]
            true_max = ebox_min_threshold * (max_val - min_val)
            for bbox_t in bboxes:
                bbox = [int(x) for x in bbox[:4]]
                if bbox_t[4] >= true_max:
                    bboxes_use.append(bbox)
                    crops = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                    crops = cv2.resize(crops, (64, 128), interpolation = cv2.INTER_CUBIC)
                    result_X.append(crops)
                X = np.asarray(result_X)
                X = X.reshape(X.shape[0], 64, 128, 1)
                predictions = model.predict(X, batch_size=12)


def main(argv):
    date = ''
    time = ''
    frame_type = 'bgrimage'
    base_path = ''
    minutes_data = ''
    if not argv:
        print (
            'Error: Not enough arguments... Usage: \npython testSUP.py --base-path /path/to/data/ --date YYYY-MM-DD ---time 0-24 --minutes 0-59 --frame bgr/depth/registered')
        sys.exit()
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["base-path=", "date=", "time=", "minutes=", "frame-type="])
        print ("opts", opts, "args", args)
    except getopt.GetoptError:
        print (
            'python testSUP.py --base-path /path/to/data/ --date YYYY-MM-DD ---time 0-24 --minutes 0-59 --frame bgr/depth/registered')
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print (
                'python testSUP.py --base-path /path/to/data/ --date YYYY-MM-DD ---time 0-24 --minutes 0-59 --frame bgr/depth/registered')
            sys.exit()
        elif opt == "--date":
            if re.match(r'[2][0][0-1][0-9]-([0][1-9]|[1][0-2])-([0][0-9]|[1-2][0-9]|[3][0-1])', arg) is None:
                print (
                    'Wrong date...')
                sys.exit()
            else:
                date = arg
        elif opt == "--time":
            if re.match(r'([0-1][0-9]|[2][0-3])', arg) is None:
                print (
                    'Wrong time...')
                sys.exit()
            else:
                time = arg
        elif opt == "--frame-type":
            if arg is None:
                frame_type = 'bgr'
            else:
                frame_type = arg
        elif opt == "--base-path":
            base_path = arg
        elif opt == "--minutes":
            if re.match(r'([0-5][0-9])', arg) is None:
                print (
                    'Wrong time...')
                sys.exit()
            else:
                minutes_data = arg
    if not minutes:
        folderpath = os.path.join(base_path, date, time)
        for minutes in os.listdir(folderpath):
            fpath = os.path.join(folderpath, minutes)
            TestData(filepath)
    else:
        fpath = os.path.join(base_path, date, time, minutes)
        TestData(filepath)


  
if __name__ == "__main__":
    main(sys.argv[1:])

