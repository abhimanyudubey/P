#include "PClassifier.hpp"

void vectorToMat(std::vector<std::vector<double> > v, cv::Mat& m){
	m = cv::Mat(v.size(),v.at(0).size(),CV_32FC1);
	for(int i=0;i<v.size();i++) for(int j=0;j<v.at(i).size();j++) m.at<float>(i,j)=v[i][j];
}

	PModel PTrain(int type, std::vector<std::vector<double> > features, std::vector<int> labels, std::vector<float> params){
		PModel out;
		switch(type){
			case 0:{
				//SVM(from opencv libsvm)
				cv::Mat XTrain;
				vectorToMat(features,XTrain);
				std::cout<<XTrain.rows<<" "<<XTrain.cols<<std::endl;

				CvSVMParams parameters;
				int svmTypeChoices[1] = {CvSVM::C_SVC};
				int kernelTypeChoices[1] = {CvSVM::LINEAR};
				int termCritChoices[1] = {CV_TERMCRIT_ITER};
				/*	We iterate among these following choices to get whatever parameters we require. The format of the vector *params* is
					hence as follows:
					params[0] = Type of SVM to be used. (Integer valued)
					params[1] = Type of Kernel to be used. (Integer valued)
					params[2] = Type of terminating criterion.
				*/
				if(params.size()==0)//No parameter entered, then assign default parameters.
				{
					params.push_back(0);
					params.push_back(0);
					params.push_back(0);
					params.push_back(100);
					params.push_back(1e-6);
				}

				parameters.svm_type = svmTypeChoices[int(params.at(0))];
				parameters.kernel_type = kernelTypeChoices[int(params.at(1))];
				parameters.term_crit = cvTermCriteria(termCritChoices[(int)params.at(2)],params.at(3),params.at(4));

				//Training the SVMs. We will train several one-vs-all SVMs.
				int bias=labels.at(0);
				std::vector<CvSVM> SVMs;
				for(int i=1;i<=labels.at(labels.size()-1)-bias;i++){
					cv::Mat isPresent(1,labels.size(),CV_32FC1);
					for(int j=0;j<labels.size();j++){
						if(labels.at(j)==i+bias) isPresent.at<float>(0,j)=1.0f; else isPresent.at<float>(0,j)=-1.0f;
					}
					CvSVM SVM;
					SVM.train(XTrain,isPresent,cv::Mat(),cv::Mat(),parameters);
					SVMs.push_back(SVM);
				}

				out.type=0;
				out.SVMs=SVMs;
				out.SVM_params = parameters;
			}
			break;
			case 1:
			//CUDA-Tree by Dan
			break;

			case 2:
			//Online incremental SVM.
			break;

			case 3:
			//MLP.
			break;
		
		}

	}
	Eigen::MatrixXf PPredict(int type, PModel model, Eigen::MatrixXf XTest){
		switch(model.type){
			case 0:{
				//SVM prediction (opencv).

			}
			break;
			case 1:
			break;
			case 2:
			break;
			case 3:
			break;
		}
	}

	PMetrics PGetMetrics(Eigen::MatrixXf YPred, Eigen::MatrixXf YTest){

	}
	void PWriteMetrics(PMetrics metrics, std::string metricsFile);
