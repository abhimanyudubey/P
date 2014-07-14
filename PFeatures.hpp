
#ifndef PFEATURES_HPP
#define PFEATURES_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Utils.hpp"
#include "PCoreset.hpp"
#include "omp.h"
#include <ctime>
#include <math.h>
#include <vector>
extern "C" {
	#include "vl/generic.h"
	#include "vl/hog.h"
    #include "vl/sift.h"
}
using namespace cv;

#define GLOBAL_WIDTH 640
#define GLOBAL_HEIGHT 480

std::vector<double> getHOGFeatures(cv::Mat& Image, int cellSize = 8);
std::vector<double> getSIFTFeatures(cv::BOWImgDescriptorExtractor& dextract, cv::Mat& image);
cv::Mat getIntegral(cv::Mat Image);
void concatFeatures(std::vector<double>& out, std::vector<std::vector<double> >& features);
void getFeatures(int code, cv::Mat source, std::vector<std::vector<double> >& out);
std::vector<double> getGridFeatures(cv::Mat& Image, int boxw=32, int boxh=32);
void showImageWithGrid(std::string nameOfWindow, cv::Mat Image, int hdist, int vdist);
void visualizeGridFeatures(std::string nameOfWindow,std::vector<double> features);
void visualizeHOGFeatures(std::string nameOfWindow,std::vector<double> features);
void normalizeVectorArray(std::vector<std::vector<char> >& v);

#endif