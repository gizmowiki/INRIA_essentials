'''
@gizmowiki
Tuesday, April, 4th '17

Test Model for people detection in depth

Can run on both Titan X and GTX 1080

Module Requirements: opencv, cuda 8, cudnn 5.1
	source ~/.profile
	module load opencv2.4.13
	module load cuda/8.0
	module load cudnn/5.1-cuda-8.0


Requirements: filelist of either images or videos

Usage:
	import sys
	sys.path.append("/home/rpandey/people_detect")
	from testNewModel import TestData
	
	TestData(filelist_path, videos=False, predict_threshold=0.999, nms_threshold=0.5,
                 ed_bx_threshold=0.7, image_enhancement=[False, False])


*Apart from filelist_path everything is optional
*image_enhancement take two boolean list first for using gaussian blur and second for histogram equalization

'''
import sys
from testDataset import TestData
# from testNewModel import TestData
import os
import numpy as np
import scipy.io as sio
sys.path.append("/data/stars/share/people_depth/people-depth/StarsDatasets/")
from StarsDatasets.pedestrian import hyguesepfl

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



annotations_base_path = "/data/stars/user/sdas/CAD60/newjoint_positions"
offset = 10
annotation_dict_cad60 = {}
for data in ["data1", "data2", "data3", "data4"]:
        base_path = os.path.join("/data/stars/share/people_depth/people-depth/cad/", data)
        print("Now parsing for CADA data ", base_path)
        for subfolders in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, subfolders)):
                        count_images = 0
                        for item in os.listdir(os.path.join(base_path, subfolders)):
                                count_images += 1
                        count_images /= 2
                        matfile = os.path.join(annotations_base_path, subfolders, 'joint_positions.mat')
                        annotations_data = sio.loadmat(matfile)
                        for i in range(count_images):
                                imgfilename_depth = os.path.join(base_path, subfolders, 'Depth_'+str(i+1)+'.png')
                                # img_depth = cv2.imread(imgfilename_depth, 0)
                                # img_depth = img_depth.astype(np.float32)
                                # img_depth -= np.min(img_depth)
                                # img_depth /= (np.max(img_depth) - np.min(img_depth))
                                # img_depth *= 255
                                # img_depth = img_depth.astype(np.uint8)
                                # imgfilename_rgb = os.path.join(base_path, subfolders, 'RGB_'+str(i+1)+'.png')
                                # img_rgb = cv2.imread(imgfilename_rgb)
                                xmin = int(np.min(annotations_data['pos_img'][0][i]) - 2*offset)
                                ymin = int(np.min(annotations_data['pos_img'][1][i]) - (2.5*offset))
                                xmax = int(np.max(annotations_data['pos_img'][0][i]) + 2*offset)
                                ymax = int(np.max(annotations_data['pos_img'][1][i]) + 2*offset)
                                if xmin < 0:
                                        xmin = 0
                                if ymin < 0:
                                        ymin = 0
                                if xmax > 320 or xmax == 0:
                                        xmax = 320
                                if ymax > 240 or ymax == 0:
                                        ymax = 240

                                bboxa = [xmin, ymin, xmax, ymax]
                                annotation_dict_cad60[imgfilename_depth] = [bboxa]

# epfl = hyguesepfl.hyguesepfl('/data/stars/share/people_depth/people-depth/epfl/EPFL_lab/HT2016/')
epfl = hyguesepfl.hyguesepfl('/data/stars/share/people_depth/people-depth/epfl/EPFL_lab/HT2016')
annotation_dict_epfl_corridor = epfl.get_data(
    '/home/rpandey/files_chk_epfl.txt')
base_out_path = "/home/rpandey/output_cad_fit_sunday"
TestData(filelist_path='/home/rpandey/files_cad_new.txt', base_out_path=base_out_path, annotation_dict=annotation_dict_cad60, image_enhancement=[False, True], predict_threshold=0.99, nms_threshold=0.6, ed_bx_threshold=0.8)
# TestData(filelist_path='/home/rpandey/files_cad_new.txt', image_enhancement=[False, False], predict_threshold=0.5, nms_threshold=0.6, ed_bx_threshold=0.5)


