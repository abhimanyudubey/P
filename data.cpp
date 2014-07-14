#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Utils.hpp"
#include "omp.h"
#include <ctime>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

void EigenToMat(Eigen::MatrixXf& e, cv::Mat& m){
	e.transposeInPlace();
	//Taking care of row-major to column-major transformation.
	float *p = e.data();
	m = cv::Mat(e.cols(),e.rows(),CV_32FC1,p);
}

void vectorToMat(std::vector<std::vector<float> > v, cv::Mat& m){
	m = cv::Mat(v.size(),v.at(0).size(),CV_32FC1);
	for(int i=0;i<v.size();i++) for(int j=0;j<v.at(i).size();j++) m.at<float>(i,j)=v[i][j];
}

int main(int argc, char* argv[]){
	std::vector<std::vector<float> > v;
	for(int i=0;i<3;i++){
		std::vector<float> vv;
		for(int j=0;j<6;j++){
			vv.push_back(10*i+j);
		}
		v.push_back(vv);
	}
	cv::Mat m;
	vectorToMat(v,m);
	for(int i=0;i<3;i++){
		for(int j=0;j<6;j++){
			std::cout<<m.at<float>(i,j)<<" ";
		}
		std::cout<<std::endl;
	}
}