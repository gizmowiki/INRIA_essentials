import json
import cv2
import os


filepath = "/home/rpandey/output_ktp_fit_optimized/results/result.json"
result = {}

with open(filepath, 'rb') as jsonfile:
	result = json.load(jsonfile)

outpath="/home/rpandey/output_ktp_fit_optimized/images"
if not os.path.exists(outpath):
	os.makedirs(os.path.join(outpath, "true_positive"))
	os.makedirs(os.path.join(outpath, "false_positive"))

i_fp = 0
i_tp = 0
for item in sorted(result.iterkeys()):
	img = cv2.imread(item, 2)
	for bbox in result[item]["false_positive"]:
		cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		base_out_path = os.path.join(outpath, "false_positive", "{0:05d}.png".format(i_fp))
		i_fp += 1
		cv2.imrwrite(base_out_path, cropped_img)
	for bbox in result[item]["true_positive"]:
		cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		base_out_path = os.path.join(outpath, "true_positive", "{0:05d}.png".format(i_tp))
		i_tp += 1
		cv2.imrwrite(base_out_path, cropped_img)
	print ("Completed for image %s" % item)

print ("Total tp %d Total fp %d" % (i_tp, i_fp))