
#ifndef PCLASSIFIER_HPP
#define PCLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "Utils.hpp"
#include "omp.h"
#include <ctime>
#include <math.h>
#include <vector>
#include <boost/thread.hpp>

using namespace std;
using namespace Eigen;

struct PModel{
	int type;
	/* Type of Model/Classifier:
		0 - SVMs (OpenCV LibSVM)
		1 - CUDA-Tree (Dan)
		2 - Online Incremental SVM <Will take in an existing SVM as a parameter>
		3 - MLP (?)
	*/
	//The model parameters are optional and overloaded.
	std::vector<CvSVM> SVMs;
	CvSVMParams SVM_params;
};

struct PMetrics{

};

void vectorToMat(std::vector<std::vector<double> > v, cv::Mat& m);
PModel PTrain(int type, std::vector<std::vector<double> > features, std::vector<int> labels, std::vector<float> params = std::vector<float>());
Eigen::MatrixXf PPredict(int type, PModel model, Eigen::MatrixXf XTest);
PMetrics PGetMetrics(Eigen::MatrixXf YPred, Eigen::MatrixXf YTest);
void PWriteMetrics(PMetrics metrics, std::string metricsFile);

#endif