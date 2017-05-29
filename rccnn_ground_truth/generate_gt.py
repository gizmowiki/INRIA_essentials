import os
import json


def generate_json(base_path):
    json_files = []
    json_dicts = {}
    for folder in os.listdir(base_path):
        if folder.startswith('output-'):
            f_data = folder.split('-')
            if folder.split('-')[1] not in json_files:
                json_files.append(f_data[1])
                json_dicts[f_data[1]] = {}
                print("Now parsing for data ", f_data[1])
            key_video = f_data[2].split('_')[0]
            key_key_video = key_video + "_depth"
            json_dicts[f_data[1]][key_video] = {}
            json_dicts[f_data[1]][key_video][key_key_video] = {}
            detection_folder = os.path.join(base_path, folder, 'detection_csv')
            for detections in os.listdir(detection_folder):
                frame_no = int(detections[4:].split('.')[0])
                max_accuracy = 0
                for lines in open(os.path.join(detection_folder, detections)):
                    lines = lines.strip()
                    bboxes = lines.split(',')
                    if float(bboxes[4]) > max_accuracy:
                        json_dicts[f_data[1]][key_video][key_key_video][frame_no] = [float(bboxes[4]), bboxes[0:4]]
            print ("Completed video folder ", folder)

    if not os.path.exists(os.path.join(base_path, 'ground_truth_rcnn')):
        os.makedirs(os.path.join(base_path, 'ground_truth_rcnn'))
    for data_type in json_dicts.keys():
        json_file_name = os.path.join(base_path, 'ground_truth_rcnn', data_type+'.json')
        with open(json_file_name, 'w') as jsonfile:
            json.dump(json_dicts[data_type], jsonfile)

generate_json('/data/stars/share/people_depth/people-depth/chalearn/output/')