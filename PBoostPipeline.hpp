
#ifndef PBOOSTPIPELINE_HPP
#define PBOOSTPIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Utils.hpp"
#include "PFeatures.hpp"
#include "PClassifier.hpp"
#include "libfreenect.hpp"
#include "omp.h"
#include <ctime>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <boost/thread.hpp>
#include <pthread.h>
#include <opencv2/core/core.hpp>

#define ROOT_OUTPUT_FOLDER "."

using namespace std;
using namespace Eigen;

void convertEigenToMat(Eigen::MatrixXf& e, cv::Mat& m);
void convertEigenToMat_Image(Eigen::MatrixXf& e, cv::Mat& m);
void convertMatToEigen(cv::Mat& m, Eigen::MatrixXf& e);
void convertMatToEigen_Image(cv::Mat& m, Eigen::MatrixXf& e);
void convertEigenVectorToMat(std::vector<Eigen::MatrixXf>& e, std::vector<cv::Mat>& m);
std::vector<int> getLabelMatrix(std::vector<std::vector<std::vector<double> > >& features, int bias=0);
void getKeyStatus(int& classID, bool& isBreak, bool& idChange);
void getKeyStatus(int& classID, bool& isBreak, bool& idChange, bool& train);
void linearizeFeatureVector(std::vector<std::vector<std::vector<double> > >& features, std::vector<std::vector<double> >& out);
void randomPermuteRows(Eigen::MatrixXf& matrix);
void vectorToArray(std::vector<std::vector<float> >v, float* p[]);
void vectorToMat(std::vector<std::vector<float> > v, cv::Mat& m);

class myMutex {
	public:
		myMutex() {
			pthread_mutex_init( &m_mutex, NULL );
		}
		void lock() {
			pthread_mutex_lock( &m_mutex );
		}
		void unlock() {
			pthread_mutex_unlock( &m_mutex );
		}
	private:
		pthread_mutex_t m_mutex;
};

class MyFreenectDevice : public Freenect::FreenectDevice {
	public:
		MyFreenectDevice(freenect_context *_ctx, int _index)
	 		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),
			m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false),
			m_new_depth_frame(false), depthMat(Size(640,480),CV_16UC1),
			rgbMat(Size(640,480), CV_8UC3, Scalar(0)),
			ownMat(Size(640,480),CV_8UC3,Scalar(0)){
			
			for( unsigned int i = 0 ; i < 2048 ; i++) {
				float v = i/2048.0;
				v = std::pow(v, 3)* 6;
				m_gamma[i] = v*6*256;
			}
		};
		// Do not call directly even in child
		void VideoCallback(void* _rgb, uint32_t timestamp) {
			std::cout << "RGB callback" << std::endl;
			m_rgb_mutex.lock();
			uint8_t* rgb = static_cast<uint8_t*>(_rgb);
			rgbMat.data = rgb;
			m_new_rgb_frame = true;
			m_rgb_mutex.unlock();
		};
		
		// Do not call directly even in child
		void DepthCallback(void* _depth, uint32_t timestamp) {
			std::cout << "Depth callback" << std::endl;
			m_depth_mutex.lock();
			uint16_t* depth = static_cast<uint16_t*>(_depth);
			depthMat.data = (uchar*) depth;
			m_new_depth_frame = true;
			m_depth_mutex.unlock();
		}
		
		bool getVideo(Mat& output) {
			m_rgb_mutex.lock();
			if(m_new_rgb_frame) {
				cv::cvtColor(rgbMat, output, CV_RGB2BGR);
				m_new_rgb_frame = false;
				m_rgb_mutex.unlock();
				return true;
			} else {
				m_rgb_mutex.unlock();
				return false;
			}
		}
		
		bool getDepth(Mat& output) {
				m_depth_mutex.lock();
				if(m_new_depth_frame) {
					depthMat.copyTo(output);
					m_new_depth_frame = false;
					m_depth_mutex.unlock();
					return true;
				} else {
					m_depth_mutex.unlock();
					return false;
				}
			}
	private:
		std::vector<uint8_t> m_buffer_depth;
		std::vector<uint8_t> m_buffer_rgb;
		std::vector<uint16_t> m_gamma;
		Mat depthMat;
		Mat rgbMat;
		Mat ownMat;
		myMutex m_rgb_mutex;
		myMutex m_depth_mutex;
		bool m_new_rgb_frame;
		bool m_new_depth_frame;
};


void serialPipelineDataset(std::string XTrainFile,  int treeHeight, int features = 7, std::string YTrainFile = "na");
void serialPipelineCamera(int treeHeight, int features = 7, int coresetSampleRate = 1024, float ratio = 0.25);
void serialPipelineKinect(int treeHeight, int features = 7, int coresetSampleRate = 1024, float ratio = 0.25);
void threadedPipelineKinect(int treeHeight, int features = 7, int coresetSampleRate = 1024, float ratio = 0.25);
#endif