'''
Created on Dec 20, 2016
@gizmowiki
At STARS Lab, INRIA

Code to plot HOG features of given images

**Dependencies**
1. Python 2.7+
2. __python_libraries__
    opencv, scipy, numpy
'''
import cv2
import scipy.io as sio
import numpy as np
annot=sio.loadmat("ground_truth_ground_plane.mat")
annotations=annot['t20140804_160621_00']
count=0
k=np.array([[365.80463078, 0.0 , 254.31510758],[0.0 , 365.80463337, 206.98513349], [0.0, 0.0, 1.0]])
dist=np.array([[ 0.08789588, -0.2702768 ,  0.        ,  0.        ,  0.09939543]])
aa=cv2.imread("/user/rpandey/home/inria/dataset/epfl_lab/20140804_160621_00/rgb000001.png")
h,  w = aa.shape[:2]
# print " h ", h, " w ", w
newmatrix, roi=cv2.getOptimalNewCameraMatrix(k, dist, (w,h), .8, (w,h));
# print "newmatrix", newmatrix
# dist=[ 0.08789588, -0.2702768 ,  0.        ,  0.        ,  0.09939543]
random_frames=[1,230,267,356,435,545,670,765,835]
for i in random_frames:#range(250, len(annotations[0][0])):
    img_name_rgb=""
    img_name_depth=""
    if (i+33)<10:
        img_name_rgb="/user/rpandey/home/inria/dataset/epfl_lab/20140804_160621_00/rgb00000"+str(i+33)+".png"
        img_name_depth="/user/rpandey/home/inria/dataset/epfl_lab/20140804_160621_00/depth00000"+str(i+33)+".png"
    if((i+33)<100 and (i+33)>=10):
        img_name_rgb="/user/rpandey/home/inria/dataset/epfl_lab/20140804_160621_00/rgb0000"+str(i+33)+".png"
        img_name_depth="/user/rpandey/home/inria/dataset/epfl_lab/20140804_160621_00/depth0000"+str(i+33)+".png"
    if((i+33)>=100 and (i+33)<1000):
        img_name_rgb="/user/rpandey/home/inria/dataset/epfl_lab/20140804_160621_00/rgb000"+str(i+33)+".png"
        img_name_depth="/user/rpandey/home/inria/dataset/epfl_lab/20140804_160621_00/depth000"+str(i+33)+".png"

    img_rgb=cv2.imread(img_name_rgb)
    img_depth=cv2.imread(img_name_depth)


    # img_rgb=cv2.undistort(img_rgb_undist, k, dist, None, newmatrix)
    # img_depth=cv2.undistort(img_depth_undist, k, dist, None, newmatrix)
    # x,y,w,h = roi
    # print " x ", x," y ", y," w ", w ," h ", h
    # img_rgb = img_rgb[y:y+h, x:x+w]
    # img_depth = img_depth[y:y+h, x:x+w]
    # img_rgb=img_rgb[25:-30,:]
    # img_depth=img_depth[25:-30,:]
    max_v=-1
    min_v=65537
    h, w=img_depth.shape[:2]
    print " h ", h, " w ", w

    for l in range(0,h):
        for m in range(0,w):
            if img_depth[l,m,1]<min_v:
                min_v=img_depth[l,m,1]
            if img_depth[l,m,1]>max_v:
                max_v=img_depth[l,m,1]

    # print " min ",min_v," max ",max_v
    for l in range(0,h):
        for m in range(0,w):
            # print len(img_depth[l,m])
            img_depth[l,m,0]=65535*img_depth[l,m,0]/29
            img_depth[l,m,1]=65535*img_depth[l,m,1]/29
            img_depth[l,m,2]=65535*img_depth[l,m,2]/29

    img_depth=cv2.medianBlur(img_depth, 5)
    # cv2.imshow("tem", img_rgb_temp)
    # cv2.waitKey(0);
    # print "shape ", img_rgb.shape

    # img_rgb_path="/user/rpandey/home/inria/dataset/epfl_lab/undistorted/image"+str(i)+".png"
    # img_depth_path="/user/rpandey/home/inria/dataset/epfl_lab/undistorted/depth"+str(i)+".png"
    # cv2.imwrite(img_rgb_path, img_rgb)
    # cv2.imwrite(img_depth_path, img_depth)


    noPerson=len(annotations[0][0][i][0][0])
    # print "Number of person ",noPerson
    for j in range (0, noPerson):
        # crop_rgb=img_rgb[annotations[0][0][i][0][0][j][0][1]:annotations[0][0][i][0][0][j][0][1]+annotations[0][0][i][0][0][j][0][3],annotations[0][0][i][0][0][j][0][0]:annotations[0][0][i][0][0][j][0][0]+annotations[0][0][i][0][0][j][0][2]]
        # crop_depth=img_depth[annotations[0][0][i][0][0][j][0][1]:annotations[0][0][i][0][0][j][0][1]+annotations[0][0][i][0][0][j][0][3],annotations[0][0][i][0][0][j][0][0]:annotations[0][0][i][0][0][j][0][0]+annotations[0][0][i][0][0][j][0][2]]
        # crop_depth_file="/user/rpandey/home/inria/dataset/epfl_lab/depth_cropped/depth"+str(count)+".png"
        cv2.rectangle(img_rgb_undist,(annotations[0][0][i][0][0][j][0][0],annotations[0][0][i][0][0][j][0][1]+60),(annotations[0][0][i][0][0][j][0][0]+annotations[0][0][i][0][0][j][0][2],annotations[0][0][i][0][0][j][0][1]+annotations[0][0][i][0][0][j][0][3]+60),0,thickness=2)
        cv2.rectangle(img_depth_undist,(annotations[0][0][i][0][0][j][0][0],annotations[0][0][i][0][0][j][0][1]+60),(annotations[0][0][i][0][0][j][0][0]+annotations[0][0][i][0][0][j][0][2],annotations[0][0][i][0][0][j][0][1]+annotations[0][0][i][0][0][j][0][3]+60),255,thickness=2)

        # crop_rgb_file="/user/rpandey/home/inria/dataset/epfl_lab/rgb_cropped/rgb"+str(count)+".png"
        # cv2.imwrite(crop_rgb_file,crop_rgb);
        # cv2.imwrite(crop_depth_file,crop_depth);
        # cv2.imshow("crp", crop_rgb)
        # cv2.waitKey(0)
        count=count+1
        # print "Cropped person ", count
    cv2.imshow("RGB", img_rgb_undist)
    cv2.imshow("Depth", img_depth_undist)
    cv2.waitKey(0);


print "Total Persons: ", count
