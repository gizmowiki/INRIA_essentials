from skimage import io
import os
import random
import pickle
import time
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2
import numpy as np

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        area_a = (a.ymax - a.ymin + 1) * (a.xmax - a.xmin + 1)
        return np.float((dx*dy))/np.float(area_a)
    else:
        return 0


def sample_negatives(imgname, bboxes, num_neg, neg_width, neg_height, max_overlap=0, maxattempts=1000):
    if not os.path.isfile(imgname):
        raise ValueError('The file %s does not exist.' %imgname)
    rects = []
    for bb in bboxes:
        rect = Rectangle(bb[0], bb[1], bb[2], bb[3])
        print "rect: ", rect
        rects.append(rect)

    print('Total number of bboxes = %s' %str(len(rect)))

    img = io.imread(imgname)
    sz = img.shape
    negatives = []
    checked = 1
    attempts = 0
    for i in range(num_neg):
        attempts = 0
        while (checked == 1) or (attempts < maxattempts):
            if attempts > maxattempts:
                attempts = 0
                break
            random.seed(time.time())
            if (sz[0] - neg_height - 1) < 0 or (sz[1] - neg_width - 1) < 0:
                attempts += 1
                continue
            tl_row = random.randint(0, sz[0] - neg_height - 1)
            random.seed(time.time())
            tl_col = random.randint(0, sz[1] - neg_width - 1)
            rand_rect = Rectangle(tl_col, tl_row, tl_col + neg_width, tl_row + neg_height)
            overlaps = map(lambda u: area(rand_rect, u), rects)
            if all(x <= max_overlap for x in overlaps):
                negatives.append((tl_col, tl_row, tl_col + neg_width, tl_row + neg_height))
                print(negatives)
                checked = 0
                attempts = maxattempts
            else:
                attempts += 1

    return negatives


class NegativeImage:
    negbox = []
    def __init__(self, imgname, bboxes, num_neg, neg_width, neg_height, max_overlap=0, maxattempts=1000):
        self.negbox = sample_negatives(imgname, bboxes, num_neg, neg_width, neg_height, max_overlap, maxattempts)
        return


# annotations = pickle.load(open('/run/netsop/u/sop-nas2a/vol/home_stars/rpandey/inria/dataset/hygues/EPFL_corridor/HT2016/Annotations/annotations.pkl'))
# for filename in os.listdir('/run/netsop/u/sop-nas2a/vol/home_stars/rpandey/inria/dataset/hygues/EPFL_corridor/HT2016/JPEGImages'):
#     # print "file", filename
#     annot_number = int(filename.split('.')[0])
#     neg_width = int(annotations[annot_number][0][2]) - int(annotations[annot_number][0][0])
#     neg_height = int(annotations[annot_number][0][3]) - int(annotations[annot_number][0][1])
#     negatives = sample_negatives(os.path.join('/run/netsop/u/sop-nas2a/vol/home_stars/rpandey/inria/dataset/hygues/EPFL_corridor/HT2016/JPEGImages', filename), annotations[annot_number], 10, 48, 128, 0.4)
#     img = cv2.imread(os.path.join('/run/netsop/u/sop-nas2a/vol/home_stars/rpandey/inria/dataset/hygues/EPFL_corridor/HT2016/JPEGImages', filename))
#     # print len(negatives[0])
#     for i in range(0, len(negatives)):
#         cv2.rectangle(img, (negatives[i][0], negatives[i][1]), (negatives[i][2], negatives[i][3]), 250, thickness=2)
#     plt.imshow(img)
#     plt.show()
