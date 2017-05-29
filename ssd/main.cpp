// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // for resize
#include <opencv2/highgui/highgui.hpp>

// STD
#include <vector>
#include <map>
#include <iostream>



//=================================================================================
//add your header HERE

#include "people_detector.h"

//=================================================================================

const int SCALE = 2;

int main(int argc, char **argv)
{  
//    bool write_to_out = false;
    std::string help = "This program will hep you example 1 image at a time. \n"
                       "To run this program, please type:\n"
                       "    ./SSD1IMG -m [path of model file] -i [path of image]\n"
                       "Example:\n"
                       "    ./SSD1IMG -m ./models/VGGNet/VOC0712/SSD_300x300/ -i toy.jpeg -delay 0\n"
                       "Since we will not check the existence of image, so please \n"
                       "make sure it exists before run this program.\n"
                       "Other useful options:\n"
                       "-delay X : delay X ms before display 0 is default.\n"
                       "-help    : display help\n";

    bool run_help = false;
    std::string image_path;
    string model_path;
    int delay = 0;
    for(int i=0; i<argc; i++) { 
    if(strcmp(argv[i], "-h")==0 || strcmp(argv[i], "-help")==0) { run_help = true; }
//	if(strcmp(argv[i], "-debug")==0) { debug = 1; }
    if(strcmp(argv[i], "-m")==0) { model_path = argv[i+1]; }
    if(strcmp(argv[i], "-i")==0) { image_path = argv[i+1]; }
    if(strcmp(argv[i], "-delay")==0) { delay= std::stoi(argv[i+1]); }
//    if(strcmp(argv[i], "-out")==0) { index_out_image_path = i+1; write_to_out = true;}
    }

    if (run_help)
    {
        std::cout << help << std::endl;
        return -1;
    }

//    std::string out_image_path = argv[index_out_image_path];

    //caffe load
    Detector p_detector;
//    string model_path = "/user/hunnguye/home/project/deep_learning/experiment/models/VGGNet/VOC0712/SSD_300x300";
    const string& model_file   = model_path + "/deploy.prototxt";
    const string& weights_file = model_path + "/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel";
    const string mean_file;
    const string mean_value = "104,117,123";
    p_detector.load(model_file, weights_file, mean_file, mean_value);
    cv::Mat regImg = cv::imread(image_path.c_str(), CV_LOAD_IMAGE_COLOR);

         cv::Mat resizedImg; cv::resize(regImg, resizedImg, cv::Size(), 1/double(SCALE), 1/double(SCALE));
         std::vector<vector<float> > detections = p_detector.Detect(resizedImg);
             if (detections.size() > 0){
                 for (int i = 0; i < detections.size(); ++i) {
                     const vector<float>& d = detections[i];
                     const float score = d[2];
                     const int type = d[1];


                     Rect SSDoutput;
                     if (score >= 0.9 && type ==15) {
                         SSDoutput.x=static_cast<int>(d[3] * resizedImg.cols) * SCALE ;
                         SSDoutput.y = static_cast<int>(d[4] * resizedImg.rows)* SCALE;
                         SSDoutput.width = static_cast<int>(d[5] * resizedImg.cols) * SCALE - SSDoutput.x;
                         SSDoutput.height = static_cast<int>(d[6] * resizedImg.rows) * SCALE - SSDoutput.y;

                     }
                     cv::rectangle(regImg, SSDoutput, cv::Scalar(0,0,255),2);
                 }
             }
             cv::imshow("SSD for 1 Image",regImg);
             cv::waitKey(delay);


//      if (write_to_out)
//      {
//            std::ostringstream ss;
//            ss << std::setw( 6 ) << std::setfill( '0' ) << frameNumber;
//            cv::imwrite(out_image_path + "/" + ss.str() + ".jpg",display_mat);
//      }
      
    return 0;
}
