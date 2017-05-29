from __future__ import print_function
"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'ignore()' it is checked wether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv
import sys
import copy
import cv2
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import numpy as np


base_path = ""
data_type = ""
video_file = ""
mat_file = ""
rgb_video = ""
depth_video = ""
annotations = {}
backup_annotations = {}
frame_number = 0
activate_storage = False


def initialize(g_base_path, g_data_type, g_video_file):
    global base_path
    global data_type
    global video_file
    global mat_file
    global rgb_video
    global depth_video
    global annotations
    global frame_number
    global activate_storage
    base_path = g_base_path
    data_type = g_data_type
    video_file = g_video_file
    for files in os.listdir(os.path.join(base_path, 'data',
                                        data_type, video_file)):
        if files.endswith('data.mat'):
            mat_file = files
        if files.endswith('color.mp4'):
            rgb_video = files
        if files.endswith('depth.mp4'):
            depth_video = files
    with open(os.path.join(base_path,
                           'ground_truth_check',
                           data_type+".json"), 'rb') as gtfile:
        annotations = json.load(gtfile)
    print (annotations.keys())
    # print annotations
    return


def viewVideo():
    cap_rgb = cv2.VideoCapture(os.path.join(base_path,
                                            'data', data_type,
                                            video_file,
                                            rgb_video))
    cap_depth = cv2.VideoCapture(os.path.join(base_path,
                                            'data', data_type,
                                            video_file,
                                            depth_video))
    video_dict = sio.loadmat(os.path.join(base_path, 'data',
                                          data_type, video_file,
                                          mat_file))
    num_frames = video_dict['Video']['NumFrames'][0][0][0][0]
    for i in range(num_frames):
        ret, frame_rgb = cap_rgb.read()
        ret_depth, frame_depth = cap_depth.read()
        if not ret:
            break
        cv2.rectangle(frame_depth,
                      (annotations[video_file][depth_video][str(i+1)][1][0],
                       annotations[video_file][depth_video][str(i+1)][1][1]),
                      (annotations[video_file][depth_video][str(i+1)][1][2],
                       annotations[video_file][depth_video][str(i+1)][1][3]),
                      color=(12,45,123),
                      thickness=2)
        cv2.rectangle(frame_rgb,
                      (annotations[video_file][depth_video][str(i + 1)][1][0],
                       annotations[video_file][depth_video][str(i + 1)][1][1]),
                      (annotations[video_file][depth_video][str(i + 1)][1][2],
                       annotations[video_file][depth_video][str(i + 1)][1][3]),
                      color=(12, 45, 123),
                      thickness=2)
        plot_img = np.concatenate((frame_rgb, frame_depth), axis=1)
        cv2.imshow("rgb-depth", plot_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    global annotations
    global backup_annotations
    global activate_storage
    backup_annotations = copy.deepcopy(annotations)
    if not activate_storage:
        backup_annotations = {}
    if activate_storage:
        print ("Succesfully updated the values. To undo press W and Next frame please! ")
        activate_storage = False
        annotations[video_file][depth_video][str(frame_number + 1)][1][0] = int(x1)
        annotations[video_file][depth_video][str(frame_number + 1)][1][1] = int(y1)
        annotations[video_file][depth_video][str(frame_number + 1)][1][2] = int(x2)
        annotations[video_file][depth_video][str(frame_number + 1)][1][3] = int(y2)
        with open(os.path.join(base_path,
                               'ground_truth_check',
                               data_type + ".json"), 'w') as gtfile:
            gtfile.write(json.dumps(annotations))

    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))


def toggle_selector(event):
    global activate_storage
    global annotations
    global backup_annotations
    if event.key in ['R', 'r']:
        print ("Now storing bbox result: Please drag to required area")
        activate_storage = True
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    if event.key in ['W', 'w']:
        print ("Now quitting....")
        sys.exit()
    if event.key in ['B', 'b']:
        if not backup_annotations:
            print ("No annotations to backup")
        else:
            print ("Frame no", str(frame_number+1), "Current Annotations",
                   annotations[video_file][depth_video][str(frame_number + 1)][1][0],
                   annotations[video_file][depth_video][str(frame_number + 1)][1][1],
                   annotations[video_file][depth_video][str(frame_number + 1)][1][2],
                   annotations[video_file][depth_video][str(frame_number + 1)][1][3])
            print ("Frame no", str(frame_number+1), "Backup Annotations",
                   backup_annotations[video_file][depth_video][str(frame_number + 1)][1][0],
                   backup_annotations[video_file][depth_video][str(frame_number + 1)][1][1],
                   backup_annotations[video_file][depth_video][str(frame_number + 1)][1][2],
                   backup_annotations[video_file][depth_video][str(frame_number + 1)][1][3])
            annotations[video_file][depth_video][str(frame_number + 1)][1][0] = \
            backup_annotations[video_file][depth_video][str(frame_number + 1)][1][0]
            annotations[video_file][depth_video][str(frame_number + 1)][1][1] = \
            backup_annotations[video_file][depth_video][str(frame_number + 1)][1][1]
            annotations[video_file][depth_video][str(frame_number + 1)][1][2] = \
            backup_annotations[video_file][depth_video][str(frame_number + 1)][1][2]
            annotations[video_file][depth_video][str(frame_number + 1)][1][3] = \
            backup_annotations[video_file][depth_video][str(frame_number + 1)][1][3]
            with open(os.path.join(base_path,
                                   'ground_truth_check',
                                   data_type + ".json"), 'w') as gtfile:
                gtfile.write(json.dumps(annotations))
            print ("Sucessflluy backed up last annotations")
            backup_annotations = {}

def selector_function(img, count):
    fig, current_ax = plt.subplots()
    plt.imshow(img)
    print("\n Frame %d" % count)

    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()
    return


def check_gt():
    global frame_number
    cap_depth = cv2.VideoCapture(os.path.join(base_path,
                                              'data', data_type,
                                              video_file,
                                              depth_video))
    video_dict = sio.loadmat(os.path.join(base_path, 'data',
                                          data_type, video_file,
                                          mat_file))
    num_frames = video_dict['Video']['NumFrames'][0][0][0][0]
    for i in range(num_frames):
        frame_number = i
        ret_depth, frame_depth = cap_depth.read()
        if not ret_depth:
            break
        cv2.rectangle(frame_depth,
                      (annotations[video_file][depth_video][str(i + 1)][1][0],
                       annotations[video_file][depth_video][str(i + 1)][1][1]),
                      (annotations[video_file][depth_video][str(i + 1)][1][2],
                       annotations[video_file][depth_video][str(i + 1)][1][3]),
                      color=(12, 45, 123),
                         thickness=2)
        selector_function(frame_depth, i+1)
    return


initialize(g_base_path='/proj/stars/data-srvpal/projects/people-depth/chalearn',
            g_data_type='test1',
            g_video_file='Sample00840')
print ("base", base_path)
# viewVideo()
check_gt()