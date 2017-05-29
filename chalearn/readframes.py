import cv2
import os
import numpy as np
import cv2

file_name = os.path.join("/home/rpandey/dataset/dataset_jan_26/chalearn/data/test1/Sample00800",
                         "Sample00800_depth.mp4")
cap = cv2.VideoCapture(file_name)
count = 0
file_path = os.path.join('/user/rpandey/home/inria', 'dataset', 'chalearn', 'depth_input')
if not os.path.exists(file_path):
    os.makedirs(file_path)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        file_name = os.path.join(file_path, '{0:06d}.jpg'.format(count))
        cv2.imwrite(file_name, frame)
        print("saved %d image to %s " % (count, file_path))
        count += 1
    else:
        break
print count
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
