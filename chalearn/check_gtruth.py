import cv2
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import numpy as np


class CheckGroundTruth:
    base_path = ""
    data_type = ""
    video_file = ""
    mat_file = ""
    rgb_video = ""
    depth_video = ""
    annotations = {}
    frame_number = 0
    activate_storage = False
    def __init__(self, base_path, data_type, video_file):
        self.base_path = base_path
        self.data_type = data_type
        self.video_file = video_file
        for files in os.listdir(os.path.join(base_path, 'data',
                                            data_type, video_file)):
            if files.endswith('data.mat'):
                self.mat_file = files
            if files.endswith('color.mp4'):
                self.rgb_video = files
            if files.endswith('depth.mp4'):
                self.depth_video = files
        with open(os.path.join(base_path,
                               'ground_truth',
                               data_type+".json"), 'rb') as gtfile:
            self.annotations = json.load(gtfile)
        # print self.annotations.keys()
        # print self.annotations
        return

    def viewVideo(self):
        cap_rgb = cv2.VideoCapture(os.path.join(self.base_path,
                                                'data', self.data_type,
                                                self.video_file,
                                                self.rgb_video))
        cap_depth = cv2.VideoCapture(os.path.join(self.base_path,
                                                'data', self.data_type,
                                                self.video_file,
                                                self.depth_video))
        video_dict = sio.loadmat(os.path.join(self.base_path, 'data',
                                              self.data_type, self.video_file,
                                              self.mat_file))
        num_frames = video_dict['Video']['NumFrames'][0][0][0][0]
        for i in range(num_frames):
            ret, frame_rgb = cap_rgb.read()
            ret_depth, frame_depth = cap_depth.read()
            if not ret:
                break
            cv2.rectangle(frame_depth,
                          (self.annotations[self.video_file][self.depth_video][str(i+1)][1][0],
                           self.annotations[self.video_file][self.depth_video][str(i+1)][1][1]),
                          (self.annotations[self.video_file][self.depth_video][str(i+1)][1][2],
                           self.annotations[self.video_file][self.depth_video][str(i+1)][1][3]),
                          color=(12,45,123),
                          thickness=2)
            cv2.rectangle(frame_rgb,
                          (self.annotations[self.video_file][self.depth_video][str(i + 1)][1][0],
                           self.annotations[self.video_file][self.depth_video][str(i + 1)][1][1]),
                          (self.annotations[self.video_file][self.depth_video][str(i + 1)][1][2],
                           self.annotations[self.video_file][self.depth_video][str(i + 1)][1][3]),
                          color=(12, 45, 123),
                          thickness=2)
            plot_img = np.concatenate((frame_rgb, frame_depth), axis=1)
            cv2.imshow("rgb-depth", plot_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        list_annot = [self.current_image, int(x1), int(y1), int(x2), int(y2)]
        print (list_annot)

        if self.activate_storage:
            print ("Next frame please! ")
            self.activate = False
            self.annotations[self.video_file][self.depth_video][str(self.frame_number + 1)][1][0] = int(x1)
            self.annotations[self.video_file][self.depth_video][str(self.frame_number + 1)][1][1] = int(y1)
            self.annotations[self.video_file][self.depth_video][str(self.frame_number + 1)][1][2] = int(x2)
            self.annotations[self.video_file][self.depth_video][str(self.frame_number + 1)][1][3] = int(y2)

        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    @staticmethod
    def toggle_selector(event):
        if event.key in ['R', 'r']:
            print ("Now storing bbox result: Please drag to required area")
            AnnotateImage.activate_storage = True
        # if event.key in []
        if event.key in ['Q', 'q'] and AnnotateImage.toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            AnnotateImage.toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not AnnotateImage.toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            AnnotateImage.toggle_selector.RS.set_active(True)
        if event.key in ['W', 'w']:
            AnnotateImage.write_annotations(AnnotateImage.output_path)

    def check_gt(self):
        cap_depth = cv2.VideoCapture(os.path.join(self.base_path,
                                                  'data', self.data_type,
                                                  self.video_file,
                                                  self.depth_video))
        video_dict = sio.loadmat(os.path.join(self.base_path, 'data',
                                              self.data_type, self.video_file,
                                              self.mat_file))
        num_frames = video_dict['Video']['NumFrames'][0][0][0][0]
        for i in range(num_frames):
            self.frame_number = i
            ret_depth, frame_depth = cap_depth.read()
            if not ret_depth:
                break
            cv2.rectangle(frame_depth,
                          (self.annotations[self.video_file][self.depth_video][str(i + 1)][1][0],
                           self.annotations[self.video_file][self.depth_video][str(i + 1)][1][1]),
                          (self.annotations[self.video_file][self.depth_video][str(i + 1)][1][2],
                           self.annotations[self.video_file][self.depth_video][str(i + 1)][1][3]),
                          color=(12, 45, 123),
                          thickness=2)
            plt.imshow(frame_depth)
            self.toggle_selector.RS = RectangleSelector(current_ax, self.line_select_callback,
                                                        drawtype='box', useblit=True,
                                                        button=[1, 3],  # don't use middle button
                                                        minspanx=5, minspany=5,
                                                        spancoords='pixels',
                                                        interactive=True)
            plt.connect('key_press_event', self.toggle_selector)
            plt.show()


        return


chk = CheckGroundTruth(base_path='/proj/stars/data-srvpal/projects/people-depth/chalearn',
                       data_type='test1',
                       video_file='Sample00840')
chk.viewVideo()