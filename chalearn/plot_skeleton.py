"""
Created on Feb 2, 2017
@gizmowiki
At STARS Lab, inria

Plot Chalearn dataset
e.g. usage:
from plot_skeleton import PlotSkeleton

plt = PlotSkeleton('/proj/stars/data-srvpal/projects/people-depth/chalearn/data/validation1/Sample00410')
plt.plot()

**Dependencies**
1. Python 2.7+
2. __python_libraries__
    os, opencv, scipy.io

"""
import os
import cv2
import scipy.io as sio
import gzip
import json
import numpy as np


class PlotSkeleton:
    mat_file = ""
    base_path = ""
    rgb_video = ""
    depth_video = ""
    type_data = ""
    video_folder = ""
    annotation_dict = {}

    def __init__(self, base_path, type_data, video_folder):
        self.base_path = base_path
        self.type_data = type_data
        self.video_folder = video_folder
        for files in os.listdir(os.path.join(self.base_path, type_data, video_folder)):
            if files.endswith('data.mat'):
                self.mat_file = files
            if files.endswith('color.mp4'):
                self.rgb_video = files
            if files.endswith('depth.mp4'):
                self.depth_video = files
        return

    def plot(self, depth_images=False):
        video_dict = sio.loadmat(os.path.join(self.base_path, self.type_data, self.video_folder, self.mat_file))
        num_frames = video_dict['Video']['NumFrames'][0][0][0][0]
        if not depth_images:
            cap = cv2.VideoCapture(os.path.join(self.base_path, self.type_data, self.video_folder, self.rgb_video), 2)
        else:
            cap = cv2.VideoCapture(os.path.join(self.base_path, self.type_data, self.video_folder, self.depth_video))
        for i in range(0, num_frames):
            ret, frame = cap.read(2)
            if not ret:
                break
            print("max", np.max(frame), "min", np.min(frame), frame.shape)
            skels = video_dict['Video']['Frames'][0][0][0][i][0][0]['PixelPosition'][0]
            for j in range(0, len(skels)):
                cv2.putText(frame, str(j+1), (skels[j][0], skels[j][1]), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3,
                            color=(0, 0, 255))
            cv2.imshow('skeleton', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return

    def read_annotations(self, annotations_path):
        filename = os.path.join(self.base_path,
                                self.type_data,
                                self.video_folder,
                                self.rgb_video)
        annot_filename = self.type_data + "-" + self.rgb_video[:-4]+".gz"
        with gzip.open(os.path.join(annotations_path, annot_filename), 'rb') as comp_annots:
            file_content = comp_annots.read()

        annotations = file_content.split('\n')
        annotations = [x.split(' ') for x in annotations]
        annotations = [(x[0], int(x[1]), float(x[2]), int(x[3]), int(x[4]), int(x[5]), int(x[6])) for x in annotations]
        annotation_dict = {}
        for annots in annotations:
            frame_num = int((annots[0].split('/')[-1]).split('.')[0])
            if frame_num not in annotation_dict.keys():
                annotation_dict[frame_num] = (0, [])
            if annots[1] == 15:
                if annots[2] > annotation_dict[frame_num][0]:
                    annotation_dict[frame_num][0] = annots[2]
                    annotation_dict[frame_num][1].append(annots[3])
                    annotation_dict[frame_num][1].append(annots[4])
                    annotation_dict[frame_num][1].append(annots[5])
                    annotation_dict[frame_num][1].append(annots[6])

        return annotation_dict
        # self.annotation_dict = annotation_dict
    def calc_full_annotations(self, annotations_path):
        for types in os.listdir(self.base_path):
            self.type_data = types
            self.annotation_dict[types] = {}
            for vidfolder in os.listdir(os.path.join(self.base_path, types)):
                self.annotation_dict[types][vidfolder] = {}
                self.video_folder = vidfolder
                for files in os.listdir(os.path.join(self.base_path, types, vidfolder)):
                    if files.endswith('data.mat'):
                        self.mat_file = files
                    if files.endswith('color.mp4'):
                        self.rgb_video = files
                    if files.endswith('depth.mp4'):
                        self.depth_video = files
                    self.annotation_dict[types][vidfolder][self.depth_video] = self.read_annotations(annotations_path)
                    print ("Completed data type: %s Video: %s " % (self.type_data, self.video_folder))
        return
    def save_json(self):
        output_json_filename = os.path.join(self.base_path,
                                            "annotations_chalearn.json")
        with open(output_json_filename, 'w') as jsonfile:
            json.dump(self.annotation_dict, jsonfile)
