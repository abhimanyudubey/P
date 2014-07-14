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

std::vector<double> getHOGFeatures(cv::Mat& Image, int cellSize = 8){
	VlHog *hog = vl_hog_new(VlHogVariantDalalTriggs, 4, VL_FALSE);
	cv::Mat img2;
	Image.convertTo(img2, CV_32FC1);
	vl_hog_put_image(hog,(float*)img2.data,Image.cols,Image.rows,1,8);
	int hogw = vl_hog_get_width(hog);
	int hogh = vl_hog_get_height(hog);
	int hogd = vl_hog_get_dimension(hog);
	float* hogArray = (float*)vl_malloc(hogw*hogh*hogd*sizeof(float));
	vl_hog_extract(hog,hogArray);
	vl_hog_delete(hog);
	cv::Mat out = cv::Mat(hogw*hogd*hogh,1,sizeof(float),hogArray);
    std::vector<double> output_vector;
    for(int i=0;i<out.rows;i++) output_vector.push_back(out.at<float>(i,0));
	return output_vector;
}

// cv::BOWImgDescriptorExtractor getSIFTBOW(std::vector<cv::Mat>& Images, int hist_dim = 256){
//     std::vector<cv::Mat> descriptors;
//     cv::SiftFeatureDetector detector(0.05,5.0);
//     cv::SiftDescriptorExtractor extractor(3.0);
//     cv::DescriptorMatcher matcher;
//     cv::BOWImgDescriptorExtractor dextract(extractor,matcher);
//     for(int i=0;i<Images.size();i++){
//         std::vector<cv::KeyPoint> kp;
//         detector.detect(Images.at(i),kp);
//         cv::Mat descriptor;
//         extractor.compute(Images.at(i),kp,descriptor);
//         descriptors.push_back(descriptor);
//     }
    
//     cv::BOWKMeansTrainer bowt(hist_dim);
//     for(int i=0;i<descriptors.size();i++) bowt.add(descriptors.at(i));
//     cv::Mat vocab = bowt.cluster();
//     dextract.setVocabulary(vocab);
//     return dextract;
// }

std::vector<double> getSIFTFeatures(cv::BOWImgDescriptorExtractor& dextract, cv::Mat& image){
    cv::SiftFeatureDetector detector(0.05,5.0);
    cv::Mat descriptor;
    std::vector<cv::KeyPoint> kp;
    detector.detect(image,kp);
    std::vector<std::vector<int> > keypointIdx;
    dextract.compute(image,kp,descriptor, &keypointIdx);
    std::vector<double> feature_vector;
    for(int i=0;i<keypointIdx.size();i++){
        double val = keypointIdx.at(i).size();
        feature_vector.push_back(val);
    }
    return feature_vector;
}

cv::Mat getIntegral(cv::Mat Image){
    cv::Mat output(Image.rows,Image.cols,CV_64FC1);
    double sum=0;
    for(int i=0;i<Image.rows;i++){
        for(int j=0;j<Image.cols;j++){
            sum+=Image.at<double>(i,j);
            output.at<double>(i,j)=sum;
        }
    }
    return output;
}


std::vector<double> getGridFeatures(cv::Mat& Image, int boxw=32, int boxh=32){
    int nBoxes = (static_cast<int>(ceil(Image.rows/boxw)))*(static_cast<int>(ceil(Image.cols/boxh)));
    #ifdef DEBUG
        std::cout<<"Total number of boxes per channel= "<<ceil(Image.rows/boxw)<<" * "<<ceil(Image.cols/boxh)<<" = "<<nBoxes<<endl;
    #endif
    std::vector<double> output_vector;
    std::vector<cv::Mat> img_channels;
    cv::split(Image,img_channels);
    for(int channel=0;channel<3;channel++){
        cv::Mat intImageold(Image.rows+1,Image.cols+1,CV_64FC1);
        double minval,maxval;
        intImageold = getIntegral(img_channels.at(channel));
        cv::Mat intImage;
        cv::normalize(intImageold,intImage,0,255,NORM_MINMAX,CV_32FC1);
        std::vector<double> sum_vector,feature_vector;
        for(int i=boxw-1;i<Image.rows;i+=boxw){
            for(int j=boxh-1;j<Image.cols;j+=boxh){
                double pixel_value = intImage.at<float>(i,j)+intImage.at<float>(i-boxw+1,j-boxh+1)-intImage.at<float>(i-boxw+1,j)-intImage.at<float>(i,j-boxh+1);
                //Getting the integral value of the square bounded by (i-1)(j-1)th and (i)(j)th of the boxes.
                sum_vector.push_back(pixel_value);
            }
        }
        for(int i=0;i<sum_vector.size();i++){
            for(int j=0;j<sum_vector.size();j++){
                double val_diff = (sum_vector.at(i) - sum_vector.at(j));
                output_vector.push_back(val_diff);
            }
        }
    }
    return output_vector;
}

void showImageWithGrid(std::string nameOfWindow, cv::Mat Image, int hdist, int vdist){
    for(int i=0;i<Image.rows;i++){
        for(int j=0;j<Image.cols;j++){
            if(i%hdist==0 || j%vdist ==0){
                Image.at<cv::Vec3f>(i,j)={0,0,0};
                }   
        }
    }
    cv::imshow(nameOfWindow,Image);
}

void visualizeGridFeatures(std::string nameOfWindow,std::vector<double> features){
    int x = features.size()/3;
    int nhor = sqrt(x);
    cv::Mat temp(nhor,features.size()/(nhor*3),CV_32FC3);
    vector<cv::Mat> channels;
    for(int k=0;k<3;k++){
        int m = sqrt(nhor);
        cv::Mat ch(nhor,features.size()/(nhor*3),CV_32FC1);
        for(int i=0;i<nhor;i++){
            for(int j=0;j<nhor;j++){
                ch.at<float>(i,j)= (0);
            }
        }
        for(int i=0;i<nhor;i++){
            for(int j=0;j<nhor;j++){
                ch.at<float>(m*(i%m)+(j%m),m*floor(i/m)+floor(j/m))= features.at(i*nhor+j+features.size()*k/3.0);
            }
        }
        channels.push_back(ch);
    }
    cv::merge(channels,temp);
    double maxval,minval;
    cv::Mat temp2;
    cv::normalize(temp,temp2,0,1,NORM_MINMAX,CV_32F);
    cv::Mat outImage;
    cv::resize(temp,outImage,Size(0,0),6,6,INTER_NEAREST);
    cv::imshow(nameOfWindow,outImage);
}

void visualizeHOGFeatures(std::string nameOfWindow,std::vector<double> features){
    int x = features.size()/3;
    #ifdef DEBUG
        std::cout<<"Visualizing features... size of feature vector is "<<features.size()<<endl;
    #endif
    int nhor = sqrt(x);
    cv::Mat temp(nhor,features.size()/(nhor*3),CV_32FC3);
    vector<cv::Mat> channels;
    for(int k=0;k<3;k++){
        cv::Mat ch(nhor,features.size()/(nhor*3),CV_32FC1);        
        for(int i=0;i<nhor;i++){
            for(int j=0;j<nhor;j++){
                ch.at<float>(i,j)= (0);
            }
        }
        for(int i=0;i<nhor;i++){
            for(int j=0;j<nhor;j++){
                ch.at<float>(i,j)= features.at(i*nhor+j+features.size()*k/3.0);
            }
        }
        channels.push_back(ch);
    }
    cv::merge(channels,temp);
    double maxval,minval;
    cv::Mat temp2;
    cv::normalize(temp,temp2,0,1,NORM_MINMAX,CV_32F);
    cv::Mat outImage;
    cv::resize(temp,outImage,Size(0,0),5,5,INTER_NEAREST);
    cv::imshow(nameOfWindow,outImage);
}

int main(int c, char** argv){
	VideoCapture cap(0); 
    if(!cap.isOpened())
        return -1;

    cv::namedWindow("image_grid",WINDOW_AUTOSIZE);
    cv::namedWindow("features",WINDOW_AUTOSIZE);
    cv::namedWindow("hogfeatures",WINDOW_AUTOSIZE);
    for(;;){
        Mat frame2;
        cap >> frame2; // get a new frame from camera
        #ifdef DEBUGIMAGE
            frame2 = imread("./input.jpg",CV_LOAD_IMAGE_COLOR);
        #endif
        cv::Mat frame;
        cv::normalize(frame2,frame,0,1,NORM_MINMAX,CV_32FC3);
        double minval,maxval;
        minMaxLoc(frame,&minval,&maxval);
        frame = frame/maxval;
        showImageWithGrid("image_grid",frame,frame2.rows/atoi(argv[1]),frame2.cols/atoi(argv[2]));
        cv::imwrite("./tempimage.jpg",frame);
        visualizeGridFeatures("features",getGridFeatures(frame,frame2.rows/atoi(argv[1]),frame2.cols/atoi(argv[2])));
        visualizeHOGFeatures("hogfeatures",getHOGFeatures(frame,16));
        if(waitKey(20)>=0) break;
    }
    return 0;
}
