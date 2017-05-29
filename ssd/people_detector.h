#ifndef PEOPLEDETECTOR_H
#define PEOPLEDETECTOR_H

#include <opencv2/core/core.hpp>
#include <list>
#include <string>
//#include <cv.h>
#include <opencv2/opencv.hpp>
//#include <highgui.h>
#include <opencv2/highgui/highgui.hpp>

#include <caffe/caffe.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace cv;

using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
 public:
  void load(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

//  Detector();


  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

#endif // NEOSSDPEOPLEDETECTOR_H
