"""
Created on Jan 28, 2016
@gizmowiki
At STARS Lab, inria

Visualize Py-faster-RCNN results for people detection

example usage:
from plot_detection import PlotDetection

plt = PlotDetection(base_path='/user/rpandey/home/inria/dataset/chalearn/input',
                    detection_path='/user/rpandey/home/inria/dataset/chalearn/detection_csv',
                    output_path='/user/rpandey/home/output/',
                    extension='.jpg')

or if you want to just show the images and not write to files

plt = PlotDetection(base_path='/user/rpandey/home/inria/dataset/chalearn/input',
                    detection_path='/user/rpandey/home/inria/dataset/chalearn/detection_csv',
                    extension='.jpg',
                    show_image=True)

**Dependencies**
1. Python 2.7+
2. __python_libraries__
    os, opencv, numpy

"""


import os
import cv2
import sys


class PlotDetection:
    base_path = ""
    detection_path = ""
    extension = ""
    output_path = ""
    detection_dict = {}

    def __init__(self, base_path, detection_path, extension, output_path='', show_image=False):
        self.base_path = base_path
        self.detection_path = detection_path
        self.extension = extension

        self.detect()
        if show_image:
            self.show_image()
        else:
            if not output_path:
                raise ValueError("Provide Output Path")
                sys.exit()
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.output_path = output_path
            self.draw_image()
        return

    def detect(self):
        for images in os.listdir(self.detection_path):
            self.detection_dict[images.split('.')[0] + self.extension] = []
            for lines in open(os.path.join(self.detection_path, images)):
                lines = lines.strip()
                lines = lines.split(',')
                self.detection_dict[images.split('.')[0] + self.extension].append(lines)
        return

    def draw_image(self):
        for images in sorted(self.detection_dict.iterkeys()):
            img = cv2.imread(os.path.join(self.base_path, images))
            for points in self.detection_dict[images]:
                cv2.rectangle(img, (int(points[0]), int(points[1])), (int(points[2]), int(points[3])), (255, 0, 0),
                              thickness=2)
            cv2.imwrite(os.path.join(self.output_path, images), img)
            print("Successfully wrote image %s on directory %s " % (images, self.output_path))
        return

    def show_image(self):
        for images in sorted(self.detection_dict.iterkeys()):
            img = cv2.imread(os.path.join(self.base_path, images))
            for points in self.detection_dict[images]:
                cv2.rectangle(img, (int(points[0]), int(points[1])), (int(points[2]), int(points[3])), (255, 0, 0),
                              thickness=2)
            cv2.imshow('Py-faster-RCNN', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print("Successfully displayed image %s " % images)
        return
