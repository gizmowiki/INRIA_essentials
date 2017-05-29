%{
Created on Dec 21, 2016
@gizmowiki
At STARS Lab, INRIA
Code to plot HOG features of given images

**Dependencies**
1. Octave
2. __octave_libraries
  image
%}

pkg load image;
close all;
clear all;
START = 0;
END = 99;
IN_PATH = '/user/rpandey/home/inria/dataset/mensa_seq0_1.1/depthRecovered1/';
PREFIX = 'seq0_';
SUFFIX = '_0.pgm';
OUT_PATH = '/user/rpandey/home/inria/dataset/mensa_seq0_1.1/edge550/';

%n = 206;  % for testing the relative th depth

%==============================================================================================
% The rest code should work without any changes
% ==============================================================================================
for n = 11 : 11
    path = [IN_PATH PREFIX sprintf('%04d',n) SUFFIX];
    image = imread(path);
    image = imrotate(image, 90);
    image=medfilt2(image);
    edge1=edge(image, 'roberts');
    out = [OUT_PATH PREFIX sprintf('%04d',n) '_0.pgm'];
    imshow(edge1);
    imwrite(edge1, out);
    k = waitforbuttonpress;
    disp(['Wrote to: ' out]);
end
