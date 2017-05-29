from __future__ import print_function
"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked wether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv
import sys


class AnnotateImage:
    ground_truth_list = []
    input_path = ""
    output_path = ""
    current_image = ""
    extension = ""
    image_count = 100000
    activate_storage = False

    def __init__(self, input_path, extension, image_count=10000, output_path=''):
        self.input_path = input_path
        self.extension = extension
        self.image_count = image_count
        if output_path == '':
            self.output_path = input_path
        else:
            self.output_path = output_path
        print ("lalalal", self.output_path)
        self.draw_img()
        return

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        list_annot = [self.current_image, int(x1), int(y1), int(x2), int(y2)]
        print (list_annot)

        if self.activate_storage:
            print ("Next frame please! ")
            self.ram = False
            self.ground_truth_list.append(list_annot)

        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    @staticmethod
    def toggle_selector(event):
        if event.key in ['R', 'r']:
            print ("Now storing bbox result: Please drag to required area")
            AnnotateImage.activate_storage = True
        if event.key in ['Q', 'q'] and AnnotateImage.toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            AnnotateImage.toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not AnnotateImage.toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            AnnotateImage.toggle_selector.RS.set_active(True)
        if event.key in ['W', 'w']:
            AnnotateImage.write_annotations(AnnotateImage.output_path)
            sys.exit()

    def selector_function(self, img, count):
        fig, current_ax = plt.subplots()
        plt.imshow(img)
        print("\n Frame %d" % count)

        # drawtype is 'box' or 'line' or 'none'
        self.toggle_selector.RS = RectangleSelector(current_ax, self.line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        plt.connect('key_press_event', self.toggle_selector)
        plt.show()
        return

    def draw_img(self):
        count = 0
        for files in os.listdir(self.input_path):
            count += 1
            if count == self.image_count:
                break
            if files.endswith(self.extension):
                img = mpimg.imread(os.path.join(self.input_path, files))
                self.current_image = os.path.join(self.input_path, files)
                self.selector_function(img, count)

        return

    @classmethod
    def write_annotations(self, output_path):
        file_name = os.path.join(AnnotateImage.output_path, 'annotations.txt')
        print ("Now writing in %s" % file_name)
        with open(file_name, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for item in self.ground_truth_list:
                spamwriter.writerow(item)
        return

ab = AnnotateImage(
                    input_path='/user/rpandey/home/inria/epfl_lab/20140804_160621_00/',
                    extension='.png', image_count=200,
                    output_path='/user/rpandey/home/')
