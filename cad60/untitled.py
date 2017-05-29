import os
import scipy.io as sio
import cv2
import numpy as np


annotations_base_path = "/data/stars/user/sdas/CAD60/newjoint_positions"

write_base_path = "/data/stars/share/people_depth/people-depth/fulldata"

offset = 10

crop_id = 1785306

for data in ["data1", "data2", "data3", "data4"]:
	base_path = os.path.join("/data/stars/share/people_depth/people-depth/cad/", data)
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
				img_depth = cv2.imread(imgfilename_depth)
				img_depth = img_depth.astype(np.float32)
				img_depth -= np.min(img_depth)
				img_depth /= (np.max(img_depth) - np.min(img_depth))
				img_depth *= 255
				img_depth = img_depth.astype(np.uint8)
				imgfilename_rgb = os.path.join(base_path, subfolders, 'RGB_'+str(i+1)+'.png')
				img_rgb = cv2.imread(imgfilename_rgb)
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
				crop_img = img_depth[ymin:ymax, xmin:xmax]
				crop_img = cv2.resize(crop_img, (64, 128))
				filename = os.path.join(write_base_path, 'positives', '{0:08d}.jpg'.format(crop_id))
				cv2.imshow("",crop_img)
				cv2.imwrite(filename, crop_img)
				print("Completed writing people depth in", filename)
				crop_id += 1
				# cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0,255,0))
				# cv2.rectangle(img_depth, (xmin, ymin), (xmax, ymax), (255,255,255))
				# vis = np.concatenate((img_rgb, img_depth), axis=1)
				# cv2.imshow("rgb-depth", vis)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
