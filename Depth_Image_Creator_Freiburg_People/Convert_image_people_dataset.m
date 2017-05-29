%CONFIG
% change these vars to adapt with your config.
pkg load image;
START = 0;
END = 1132;
IN_PATH = '/user/rpandey/home/inria/dataset/mensa_seq0_1.1/depth/';
PREFIX = 'seq0_';
SUFFIX = '_0.pgm';
OUT_PATH = '/user/rpandey/home/inria/dataset/mensa_seq0_1.1/depthRecovered1/';

%n = 206;  % for testing the relative th depth

%==============================================================================================
% The rest code should work without any changes
%==============================================================================================
for n = START : END
    path = [IN_PATH PREFIX sprintf('%04d',n) SUFFIX];
    image = imread(path);
    image = swapbytes(image);
    image = bitand(image,2047);
    % image = imrotate(image, 90);
    image = double(image);
    image = 1000 ./(-0.00307 .* image + 3.33);%magic formula to convert disparity to real depth
    image = uint16(image);
    out = [OUT_PATH PREFIX sprintf('%04d',n) '_0.pgm'];
    imwrite(image, out);
    disp(['Wrote to: ' out]);
end
