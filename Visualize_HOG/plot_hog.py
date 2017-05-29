'''
Created on Dec 21, 2016
@gizmowiki
At STARS Lab, INRIA

Code to plot HOG features of given images

**Dependencies**
1. Python 2.7+
2. __python_libraries__
    matplotlib,opencv,scikit-image,cython(for scikit-image), sys

**Usage**
python plot_hog.py <input_RGB_file_path> <input_depth_file_path> <x> <y> <w> <h>
'''


import matplotlib.pyplot as plt
import cv2
import sys
from skimage.feature import hog
from skimage import data, color, exposure
import random

# print type(data.astronaut())
if len(sys.argv)!=7:
    raise NameError(' Python usage:\n python plot_hog.py <input_RGB_file_path> <input_depth_file_path> <x> <y> <w> <h>')
# templist=[]
# for i in range(0,20):
#     templist.append(random.randint(0,140))
# for i in templist:
argumentsList=sys.argv
print argumentsList
for i in range(3,7):
    argumentsList[i]=int(argumentsList[i])
full_img=cv2.imread(argumentsList[2])

img=full_img[argumentsList[4]:argumentsList[4]+argumentsList[6],argumentsList[3]:argumentsList[3]+argumentsList[5]]
image = color.rgb2gray(img)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
# print "fd", fd.shape
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input Depth')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG')
ax1.set_adjustable('box-forced')


full_rimg=cv2.imread(argumentsList[1])
rimg=full_rimg[argumentsList[4]:argumentsList[4]+argumentsList[6],argumentsList[3]:argumentsList[3]+argumentsList[5]]
rimage = color.rgb2gray(rimg)
rfd, rhog_image = hog(rimage, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

# rfig, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax3.axis('off')
ax3.imshow(rimage, cmap=plt.cm.gray)
ax3.set_title('Input RGB')
ax1.set_adjustable('box-forced')

rhog_image_rescaled = exposure.rescale_intensity(rhog_image, in_range=(0, 0.02))

ax4.axis('off')
ax4.imshow(rhog_image_rescaled, cmap=plt.cm.gray)
ax4.set_title('RGB HOG')
ax1.set_adjustable('box-forced')

plt.show()
