import os
import numpy as np
import cv2


base_path = '/data/stars/share/people_depth/people-depth/ktp/'
filelist_path = '/data/stars/share/people_depth/people-depth/ktp/filelist.txt'
filelist_gt = '/data/stars/share/people_depth/people-depth/ktp/filelist_gt.txt'
annotations_dict = {}

for gt_d in open(filelist_gt, 'rb'):
    gt_d = gt_d.strip()
    bs_name = gt_d.split('/')[-1].split('_')[0]
    for g_truth in open(gt_d, 'rb'):
        g_truth = g_truth.strip()
        img_name = g_truth.split(':')[0] + '.pgm'
        key_d = os.path.join(base_path, 'images', bs_name, 'depth', img_name)
        if key_d not in annotations_dict.keys():
            annotations_dict[key_d] = []
        if g_truth.split(':')[-1]:
            bboxes = g_truth.split(':')[-1].split(',')[:-1]
            for bb in bboxes:
                bboxxx = [int(x) for x in  bb.split(' ')[2:]]
                bboxxx[3] += bboxxx[1]
                bboxxx[2] += bboxxx[0]
                annotations_dict[key_d].append(bboxxx)


i = 0
base_out_path = "/local/people_depth/"
for img_file in sorted(annotations_dict.iterkeys()):
    img = cv2.imread(img_file, 2)
    img = img.astype(np.float32)
    img /= img.max()
    img *= 65535
    img = img.astype(np.uint16)
    for bbox in annotations_dict[img_file]:
        crop_img = img[bbox[1]:bbox[3], bbox[0]: bbox[2]]
        filename = os.path.join(base_out_path, 'positives', 'ktp_{0:08d}.png'.format(i))
        cv2.imwrite(filename, crop_img)
        print ("Successfully written to ", filename)
        i += 1
    # cv2.imshow("", img)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break