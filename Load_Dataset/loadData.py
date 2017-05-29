'''
Created on Dec 20, 2016
@gizmowiki
At STARS Lab, inria

Load Dataset into memory with bounding box based on annotations
input: dataset path, annotations file path
output: different functions to load dataset

**Dependencies**
1. Python 2.7+
2. __python_libraries__
    os, opencv, glob, sys, numpy, scipy.io
'''


# import glob
# for filename in glob.glob('*.txt'):
#    # do your stuff

import os
import cv2
import glob
import sys
import numpy as np
import scipy.io as sio

# Class loadData to load dataset into the system
class loadData:
    positive_data_path=""
    data_type=""
    RGB_PREFIX="seq"
    DEPTH_PREFIX="seq"
    REFLECT_PREFIX="ref_"
    annotationsFilePath=""
    img_size=()
    img_depth={}
    descriptor=[]
    def __init__ (self, positive_data_path):
        if positive_data_path!="":
            self.positive_data_path=positive_data_path
        else:
            self.positive_data_path=os.getcwd()

        data_type_list=[]
        for filename in os.listdir(self.positive_data_path):
            if '.' in filename:
                if filename[filename.index('.'):] not in data_type_list:
                    data_type_list.append(filename[filename.index('.'):])
        if len(data_type_list)==1:
            self.data_type=data_type_list[0]
        else:
            print "Which type of file you want to import: \n"
            for i in range(0, len(data_type_list)):
                print "",i+1,". ",data_type_list[i]
            k=input("\n Enter Choice: ")
            self.data_type=data_type_list[k-1]

        return

    def getSize(self):
        for filename in os.listdir(self.positive_data_path):
            if '.' in filename:
                if filename[filename.index('.'):]==self.data_type:
                    filePath=os.path.join(self.positive_data_path,filename)
                    self.img_size=(cv2.imread(filePath)).shape
                    break
        return self.img_size
    # This function will display the rgb and depth image of given frame
    def showFrame(self, frameNumber):
        # fileType='*'+self.data_type
        count_rgb=0
        count_dep=0
        count=0
        fileRGBPath=""
        fileDepthPath=""
        for filename in os.listdir(self.positive_data_path):
            if '.' in filename:
                if filename[filename.index('.'):]==self.data_type:
                    count=count+1
                    if filename[0:len(self.RGB_PREFIX)]==self.RGB_PREFIX:
                        if count_rgb==frameNumber-1:
                            fileRGBPath=os.path.join(self.positive_data_path,filename)
                        count_rgb=count_rgb+1
                    if filename[0:len(self.DEPTH_PREFIX)]==self.DEPTH_PREFIX:
                        if count_dep==frameNumber-1:
                            fileDepthPath=os.path.join(self.positive_data_path,filename)
                        count_dep=count_dep+1


        if frameNumber-1<min(count_dep, count_rgb):
            img_rgb=cv2.imread(fileRGBPath)
            img_depth=cv2.imread(fileDepthPath, 2)
            cv2.imshow("RGB", img_rgb)
            cv2.imshow("DEPTH", img_depth)
            cv2.waitKey(0)
        else:
            print "Frame Number exceeds"

        return

    # This function display the given frame with annotations. Currently it supports only EPFL dataset
    def showAnnotatedFrameEPFL(self, annotationsFilePath, frameNumber):
        self.annotationsFilePath=annotationsFilePath
        count_rgb=0
        count_dep=0
        count=0
        fileRGBPath=""
        fileDepthPath=""
        if frameNumber<33:
            self.showFrame(frameNumber)
        else:
            for filename in os.listdir(self.positive_data_path):
                if '.' in filename:
                    if filename[filename.index('.'):]==self.data_type:
                        count=count+1
                        if filename[0:len(self.RGB_PREFIX)]==self.RGB_PREFIX:
                            if count_rgb==frameNumber-1:
                                fileRGBPath=os.path.join(self.positive_data_path,filename)
                            count_rgb=count_rgb+1
                        if filename[0:len(self.DEPTH_PREFIX)]==self.DEPTH_PREFIX:
                            if count_dep==frameNumber-1:
                                fileDepthPath=os.path.join(self.positive_data_path,filename)
                            count_dep=count_dep+1
            if frameNumber-1<min(count_dep, count_rgb):
                # img_depth=self.depthImageEnhance(fileDepthPath)
                img_rgb, img_depth=self.annotateImage(fileRGBPath,fileDepthPath, frameNumber)
                cv2.imshow("RGB", img_rgb)
                cv2.imshow("DEPTH", img_depth)
                cv2.waitKey(0)
            else:
                print "Frame Number exceeds"

        return
    # This function is used to annotate images of EPFL dataset
    def annotateImage(self,fileRGBPath,fileDepthPath, frameNumber):
        img_rgb=cv2.imread(fileRGBPath)
        img_depth=self.depthImageEnhance(fileDepthPath)
        annot=sio.loadmat(self.annotationsFilePath)
        annotations=annot['t20140804_160621_00']
        noPerson=len(annotations[0][0][frameNumber-33][0][0])

        for j in range (0, noPerson):
            cv2.rectangle(img_rgb,(annotations[0][0][frameNumber-33][0][0][j][0][0],annotations[0][0][frameNumber-33][0][0][j][0][1]+60),(annotations[0][0][frameNumber-33][0][0][j][0][0]+annotations[0][0][frameNumber-33][0][0][j][0][2],annotations[0][0][frameNumber-33][0][0][j][0][1]+annotations[0][0][frameNumber-33][0][0][j][0][3]+60),0,thickness=2)
            cv2.rectangle(img_depth,(annotations[0][0][frameNumber-33][0][0][j][0][0],annotations[0][0][frameNumber-33][0][0][j][0][1]+60),(annotations[0][0][frameNumber-33][0][0][j][0][0]+annotations[0][0][frameNumber-33][0][0][j][0][2],annotations[0][0][frameNumber-33][0][0][j][0][1]+annotations[0][0][frameNumber-33][0][0][j][0][3]+60),255,thickness=2)

        return img_rgb, img_depth

    # This functions apply enhance the depth image by scaling (normalizing) and removing noise (medianFilter)
    def depthImageEnhance(self,fileDepthPath):
        img_depth=cv2.imread(fileDepthPath, 2)
        # max_v=img_depth.max()
        # min_v=img_depth.min()
        # h, w=self.img_size[:2]
        #
        # for l in range(0,h):
        #     for m in range(0,w):
        #         if img_depth[l,m,1]<min_v:
        #             min_v=img_depth[l,m,1]
        #         if img_depth[l,m,1]>max_v:
        #             max_v=img_depth[l,m,1]

        # for l in range(0,h):
        #     for m in range(0,w):
        #         img_depth[l,m,0]=65535*(img_depth[l,m,0]-min_v)/(max_v-min_v)
        #         img_depth[l,m,1]=65535*(img_depth[l,m,1]-min_v)/(max_v-min_v)
        #         img_depth[l,m,2]=65535*(img_depth[l,m,2]-min_v)/(max_v-min_v)
        #
        img_depth *= (65535.0/img_depth.max())
        img_depth=cv2.medianBlur(img_depth, 5)

        return img_depth

    # This functions is under construction. This currently is designed for People dataset to compute HOG features
    def computeHOGpeopleDataset(self,fileTrackPath):

        p_count=0
        for filename in os.listdir(fileTrackPath):
            count=0
            print "\n next file ", filename
            fileTrack=os.path.join(fileTrackPath,filename)
            for line in open(fileTrack):
                line = line.strip()
                fields = str(line).split(' ')
                if count==0:
                    count=1
                    continue
                imgName=fields[0]+self.data_type
                imgPath=os.path.join(self.positive_data_path,imgName)

                self.img_depth[fields[0]]=self.depthImageEnhance(imgPath)
                # self.img_depth[fields[0]]=cv2.imread(imgPath, 2)

                # rows,cols = self.img_depth[fields[0]].shape[:2]
                # M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
                # self.img_depth[fields[0]] = cv2.warpAffine(self.img_depth[fields[0]],M,(cols,rows))
                # print fields[0]," : ",fields[2]," : ",fields[3]," : ",fields[4]," : ",fields[5]," : "
                # cv2.rectangle(self.img_depth[fields[0]],(int(fields[2]), int(fields[3])),(int(fields[2])+int(fields[4]), int(fields[3])+int(fields[5])), 255, thickness=2)
                for i in [2,3,4,5]:
                    if int(fields[i])<0:
                        fields[i]=0
                # print fields[0]," : ",fields[2]," : ",fields[3]," : ",fields[4]," : ",fields[5]," : "
                crop_img_r=self.img_depth[fields[0]][int(fields[3]):int(fields[3])+int(fields[5]),int(fields[2]):int(fields[2])+int(fields[4])]
                crop_img=self.rotateImage(crop_img_r)
                fileNameNumpy="checkDepth/numpy"+str(p_count)+".png"
                fileFullNameNumpy=os.path.join(os.getcwd(), fileNameNumpy)
                # fileNameNumpy="numpy"+str(p_count)+".npy"
                cv2.imwrite(fileFullNameNumpy, crop_img)
                # print " shape ", crop_img.shape
                crop_img=cv2.resize( crop_img, (64,128) );
                #applying HOG Descriptor
                hog = cv2.HOGDescriptor()
                #storing computed hog features in the output array in the output list
                self.descriptor.append([])
                print "shape: ",crop_img.shape, " min: ", crop_img.min(), " max: ", crop_img.max()
                crop_img = (crop_img/256).astype('uint8')
                self.descriptor[p_count].append(hog.compute(crop_img))
                hogfeatNumpy=np.array(self.descriptor)
                # np.save(fileNameNumpy, hogfeatNumpy)
                p_count=p_count+1
                print "Computed feature for person ", p_count
                # cv2.imshow(fields[0], crop_img)
                # cv2.waitKey(0)

        #storing length of total positive images
        positive_data_length=len(self.descriptor)
        print "Total computed person", positive_data_length
        #creating a zero matrix to store all positive hog features with label
        hog_feat_with_label_positive=np.zeros((3781,positive_data_length))

        print "generating hog features with label for positive data"
        #moving data from list to a matrix and also specifying label 1 to all positivce data at the last column of the matrix
        for i in range(0,positive_data_length) :

            #initializing temporary variable to move the features data
            temp_hog_array=np.array(descriptor[i])
            hog_feat=temp_hog_array[0,:,:]
            hogfeatarray=np.array(hog_feat)
            hog_feat_with_label_positive[0:3779,i]=hogfeatarray[0:3779,0]
            #print result[0:3779,0]

            #giving label 1 to all positive data
            hog_feat_with_label_positive[3780,i]=1

        fullHOGFeatfileName="HOGFeatureswithLabel.npy"
        np.save(fullHOGFeatfileName, hog_feat_with_label_positive)

        return

    # This small functions rotates the image to 90' and then flip vertically
    def rotateImage(self,image):
        rows,cols = image.shape[:2]
        # print "rows: ", rows , " cols: ", cols, " len: ", len(self.img_size)

        if len(image.shape)==2:
            result = np.zeros((cols,rows), np.uint16)
            for i in range (0, cols):
                for j in range(0, rows):
                    result[cols-i-1,rows-j-1]=image[j,i]
        else:
            result = np.zeros((cols,rows, 3), np.uint8)
            for i in range (0, cols):
                for j in range(0, rows):
                    # print " hall ", image[j,i]
                    for k in range(0, 3):
                        result[cols-i-1,rows-j-1,k]=image[j,i,k]
        # for i in range (0, cols):
        #     for j in range(0, rows):
        #         print result[i,j]
        return result


    def createReflectedDataset(self):
        count=0
        fileRGBPath=""
        fileDepthPath=""
        for filename in os.listdir(self.positive_data_path):
            if '.' in filename:
                if filename[filename.index('.'):]==self.data_type:
                    if filename[0:len(self.RGB_PREFIX)]==self.RGB_PREFIX:
                        fileRGBPath=os.path.join(self.positive_data_path,filename)
                        img_rgb=cv2.imread(fileRGBPath)
                        reflectedFileName=self.REFLECT_PREFIX+filename
                        reflectedPathRGB=os.path.join(self.positive_data_path,reflectedFileName)
                        img_rgb_flip=cv2.flip(img_rgb,0)
                        cv2.imwrite(reflectedPathRGB,img_rgb_flip)
                        count=count+1
                        print "Reflected image ",count
                    if filename[0:len(self.DEPTH_PREFIX)]==self.DEPTH_PREFIX:
                        fileDepthPath=os.path.join(self.positive_data_path,filename)
                        img_depth=cv2.imread(fileDepthPath)
                        reflectedFileName=self.REFLECT_PREFIX+filename
                        reflectedPathDepth=os.path.join(self.positive_data_path,reflectedFileName)
                        img_depth_flip=cv2.flip(img_depth,0)
                        cv2.imwrite(reflectedPathDepth,img_depth_flip)
                        count=count+1
                        print "Reflected image ",count

        return
