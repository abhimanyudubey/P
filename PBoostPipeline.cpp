#include "PBoostPipeline.hpp"
#define SHOWFEED 1
#define FEATURE_THRESH 50
#define LEARNER_THRESH 200

void EigenToMat(Eigen::MatrixXf& e, cv::Mat& m){
	e.transposeInPlace();
	//Taking care of row-major to column-major transformation.
	float *p = e.data();
	m = cv::Mat(e.cols(),e.rows(),CV_32FC1,p);
}

void convertEigenToMat_Image(Eigen::MatrixXf& e, cv::Mat& m){
	std::vector<cv::Mat> channels;
	for(int k=0;k<3;k++){
		cv::Mat tempmat = cv::Mat(e.rows(),e.cols()/3,CV_32FC1);
		for(int i=0;i<e.rows();i++){
			for(int j=0;j<e.cols()/3;j++){
				tempmat.at<float>(i,j) = e(i,j+k*e.cols()/3)*1.0;
			}
		}
		channels.push_back(tempmat);
	}
	cv::merge(channels,m);
}

void convertMatToEigen(cv::Mat& m, Eigen::MatrixXf& e){
	e.resize(m.rows,m.cols);
	//std::cout<<"Debugging"<<e.rows()<<" "<<e.cols()<<endl;
	for(int i=0;i<m.rows;i++){
		for(int j=0;j<m.cols;j++){
			e(i,j) = m.at<float>(i,j);
		}
	}
}

void convertMatToEigen_Image(cv::Mat& m, Eigen::MatrixXf& e){
	std::vector<cv::Mat> channels;
	cv::split(m,channels);
	e.resize(m.rows,m.cols*3);
	e.setZero();
	for(int i=0;i<m.rows;i++){
		for(int j=0;j<m.cols;j++){
			e.coeffRef(i,j) = channels.at(0).at<float>(i,j)*1.0;
			e.coeffRef(i,m.cols+j) = channels.at(1).at<float>(i,j)*1.0;
			e.coeffRef(i,2*m.cols+j) = channels.at(2).at<float>(i,j)*1.0;	
		}
	}
}

void convertMatToEigen_ImageDepth(cv::Mat& m, cv::Mat d, Eigen::MatrixXf& e){
	std::vector<cv::Mat> channels;
	cv::split(m,channels);
	e.resize(m.rows,m.cols*4);
	e.setZero();
	for(int i=0;i<m.rows;i++){
		for(int j=0;j<m.cols;j++){
			e.coeffRef(i,j) = channels.at(0).at<float>(i,j)*1.0;
			e.coeffRef(i,m.cols+j) = channels.at(1).at<float>(i,j)*1.0;
			e.coeffRef(i,2*m.cols+j) = channels.at(2).at<float>(i,j)*1.0;	
			e.coeffRef(i,3*m.cols+j) = d.at<float>(i,j);
		}
	}
}

void convertEigenVectorToMat(std::vector<Eigen::MatrixXf>& e, std::vector<cv::Mat>& m){
	while(e.size()>0){

	}
	for(int i=0;i<e.size();i++){
		omp_set_num_threads(e.at(i).cols());
		cv::Mat mm(e.at(i).rows(),e.at(i).cols(),CV_32FC1);
		#pragma omp parallel
		{
			int nc = omp_get_thread_num();
			for(int j=0;j<e.at(i).rows();j++){
				mm.at<float>(j,nc) = e.at(i)(j,nc);
			}
		}
		m.push_back(mm);
	}
}

std::vector<int> getLabelMatrix(std::vector<std::vector<std::vector<double> > >& features, int bias){
	std::vector<int> labels;
	for(int i=0;i<features.size();i++){
		for(int j=0;j<features.at(i).size();j++){
			labels.push_back(i+bias);
		}
	}
	return labels;
}

void getKeyStatus(int& classID, bool& isBreak, bool& idChange){
	int keyID = cv::waitKey(30);
	switch (keyID) {
		case 32:
			classID++;
			idChange=true;
			break;
		case 27:
			isBreak=true;
			break;
	}
}

void getKeyStatus(int& classID, bool& isBreak, bool& idChange, bool& train){
	int keyID = cv::waitKey(30);
	switch (keyID) {
		case 32:
			classID++;
			idChange=true;
			break;
		case 27:
			isBreak=true;
			break;
	}
}

void linearizeFeatureVector(std::vector<std::vector<std::vector<double> > >& features, std::vector<std::vector<double> >& out){
	for(int i=0;i<features.size();i++){
		for(int j=0;j<features.at(i).size();j++){
			out.push_back(features.at(i).at(j));
		}
	}
	features.erase(features.begin(),features.end());
}


void randomPermuteRows(Eigen::MatrixXf& matrix){
	for(int i=0;i<matrix.rows();i++){
		int r_1 = rand()%matrix.rows();
		int r_2 = rand()%matrix.rows();
		matrix.row(r_1)+= matrix.row(r_2);
		matrix.row(r_2)-= matrix.row(r_1);
		matrix.row(r_2)*=-1;
		matrix.row(r_1)-= matrix.row(r_2);
	}
};

void vectorToArray(std::vector<std::vector<float> >v, float* p[]){
	for(int i=0;i<v.size();i++){
		p[i] = &v.at(i)[0];
	}
}

void vectorToMat(std::vector<std::vector<float> > v, cv::Mat& m){
	m = cv::Mat(v.size(),v.at(0).size(),CV_32FC1);
	for(int i=0;i<v.size();i++) for(int j=0;j<v.at(i).size();j++) m.at<float>(i,j)=v[i][j];
}

void serialPipelineDataset(std::string XTrainFile,  int treeHeight, int features, std::string YTrainFile){
	std::vector<Eigen::SparseMatrix<float> > XTrain;
	//Container vector of sparse float matrices for training data.

	if(YTrainFile.compare("na")!=0)	pReadData(XTrainFile,YTrainFile,XTrain); else pReadSparseData(XTrainFile,XTrain);
	//If training label file is separate, data is not in sparse LibSVM format. 

	std::string ROut = XTrainFile + ".model";
	//This is the default paradigm for model storage.

	std::vector<Eigen::MatrixXf> Coresets;
	for(int i=0;i<XTrain.size();i++){
		Eigen::MatrixXf XTrain_i = MatrixXf(XTrain.at(i));
		Coresets.at(i) = pComputeCoresetTree(XTrain_i,treeHeight);
	}

	std::vector<std::vector<std::vector<double> > > ft_mat;
	for(int i=0;i<Coresets.size();i++){
		std::vector<std::vector<double> > v;
		ft_mat.push_back(v);
	}

	std::vector<cv::Mat> CoresetsMat;
	convertEigenVectorToMat(Coresets,CoresetsMat);

	omp_set_num_threads(CoresetsMat.size());
	#pragma omp parallel
	{
		int Idx = omp_get_thread_num();	
		getFeatures(features, CoresetsMat.at(Idx), ft_mat.at(Idx));
		//Getting the features for all classes in parallel.	
	}
	std::vector<std::vector<double> > ft_mat_lin;
	linearizeFeatureVector(ft_mat,ft_mat_lin);

	PModel ft_model = PTrain(0, ft_mat_lin, getLabelMatrix(ft_mat));
	//PClassifier::writeModel(ft_model,ROut);

	std::string XTestFile = XTrainFile + ".test";
	std::string YTestFile = XTrainFile + ".test.label";
	std::string metricsFile = XTrainFile + ".metrics";

	Eigen::MatrixXf XTest,YTest;
	readMatrix(XTestFile,XTest);
	readMatrix(YTestFile,YTest);

	//Eigen::MatrixXf YPred = PClassifier::Predict(0, ft_model, XTest);
	//PClassifier::Metrics metrics = PClassifier::GetMetrics(YPred, YTest);
	//PClassifier::writeMetrics(metrics,metricsFile);
}

void serialPipelineCamera(int treeHeight, int features, int coresetSampleRate, float ratio){
	cv::VideoCapture camera(0);
	if(!camera.isOpened()){
		fprintf(stderr,"No camera feed found.");
		return;
	}

	#ifdef SHOWFEED
		cv::namedWindow("Camera Feed",WINDOW_AUTOSIZE);
	#endif

	Eigen::MatrixXf sample_frames;
	float img_x = camera.get(CV_CAP_PROP_FRAME_HEIGHT)*ratio;
	float img_y = camera.get(CV_CAP_PROP_FRAME_WIDTH)*ratio;
	sample_frames.resize(coresetSampleRate,img_x*img_y*3);
	std::vector<int> YTrain;
	std::vector<int> changeLoc;
	changeLoc.push_back(0);
	bool idChange=false;
	int classID = 0;
	bool isBreak=false;

	for(int i=0;;i++){
		if(changeLoc.size()==0) changeLoc.push_back(0);
		cv::Mat camera_frame, camera_frame_resized,camera_frame_normalized;
		camera >> camera_frame;
		cv::Mat cam2;
		camera_frame.convertTo(cam2,CV_32FC3);
		cv::resize(cam2,camera_frame_resized,Size(img_y,img_x),0,0,INTER_LINEAR);
        cv::normalize(camera_frame_resized,camera_frame_normalized,0,1,NORM_MINMAX,CV_32FC3);

        #ifdef SHOWFEED
        	cv::Mat feed_image;
        	camera_frame_normalized.copyTo(feed_image);
        	ostringstream classNum;
        	classNum << changeLoc.size();
        	std::string camera_text = "Learning class number: " + classNum.str();
        	cv::putText(feed_image, camera_text, cvPoint(10,10), FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(200,200,250));
        	cv::imshow("Camera Feed",feed_image);
        #endif
        
        getKeyStatus(classID,isBreak,idChange);
        if(idChange){
        	idChange=false;
        	changeLoc.push_back(i%coresetSampleRate);
        }
        if(isBreak) break;

        if(!(i%coresetSampleRate)&i>0){
        	std::cout<<"Sampling for coreset construction #"<<floor(i/coresetSampleRate)<<"..."<<std::endl;
        	Eigen::MatrixXf XTrain = sample_frames;

        	//std::string ROut = XTrainFile + ".model";

        	int nClasses = changeLoc.size();
        	std::cout<<"Number of training classes: "<<nClasses<<endl;
        	changeLoc.push_back(coresetSampleRate-1);
        	cv::Mat XTrain_Mat;
        	//convertEigenToMat(XTrain,XTrain_Mat);

        	std::vector<Eigen::MatrixXf> Coresets;
        	for(int ii=1;ii<changeLoc.size();ii++){
        		std::cout<<"Coreset details "<<changeLoc.at(ii-1)<<" "<<changeLoc.at(ii)<<" "<<XTrain.cols()<<endl;
        		Eigen::MatrixXf XTrain_cut1 = XTrain.block(changeLoc.at(ii-1),0,changeLoc.at(ii)-changeLoc.at(ii-1),XTrain.cols());
        		Eigen::MatrixXf XTrain_cut = XTrain_cut1;
        		//randomPermuteRows(XTrain_cut);
        		Eigen::MatrixXf Ctemp = pComputeCoresetTree(XTrain_cut,treeHeight,1);
        		Coresets.push_back(Ctemp);
        	}
        	sample_frames.resize(0,0);																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																														
        	//cv::normalize(disp_mat,disp_mat_normalized,0,1,NORM_MINMAX,CV_32FC3);
        	for(int cd=0;cd<Coresets.size();cd++){
        		for(int cc=0;cc<Coresets.at(cd).rows();cc++){
        			Eigen::MatrixXf disp_image = Coresets.at(cd).row(cc);
        			disp_image.resize(img_x,img_y*3);
        			cv::Mat disp_mat,disp_mat_normalized;
        			convertEigenToMat_Image(disp_image,disp_mat);
        			double dispmin,dispmax;
        			cv::minMaxLoc(disp_mat,&dispmin,&dispmax);
        			disp_mat_normalized = (disp_mat);
        			ostringstream coresetnum;
        			coresetnum << cc;
        			coresetnum << " ";
        			coresetnum << cd;
        			std::string cctext = "Coreset : " + coresetnum.str();
        			cv::namedWindow(cctext,WINDOW_AUTOSIZE);
        			cv::imshow(cctext,disp_mat);
        		}
        	}
        	cv::waitKey(0);
        	//We are now displaying the top image in the coreset. 


			std::cout<<"Constructed coresets..."<<endl;
			std::vector<std::vector<std::vector<double> > > ft_mat;
        	std::vector<cv::Mat> CoresetsMat;
        	for(int ev=0;ev<Coresets.size();ev++){
        		cv::Mat temp;
        		EigenToMat(Coresets.at(ev),temp);
        		CoresetsMat.push_back(temp);
        	}
			for(int Idx=0;Idx<Coresets.size();Idx++){
				std::vector<std::vector<double> > t;	
				getFeatures(features, CoresetsMat.at(Idx), t);
				ft_mat.push_back(t);
			}
			CoresetsMat.erase(CoresetsMat.begin(),CoresetsMat.end());
	

			std::vector<int> labels = getLabelMatrix(ft_mat,classID);			
			std::vector<std::vector<double> > ft_mat_lin;
			linearizeFeatureVector(ft_mat,ft_mat_lin);
			PModel ft_model = PTrain(0, ft_mat_lin, labels);
			ft_mat.erase(ft_mat.begin(),ft_mat.end());
			//PClassifier::writeModel(ft_model,ROut);

        	///Executing clearing commands.
        	YTrain.erase(YTrain.begin(),YTrain.end());
        	changeLoc.erase(changeLoc.begin(),changeLoc.end());
			sample_frames.resize(coresetSampleRate,img_x*img_y*3);
        }

        else{
       		Eigen::MatrixXf frame_eigen;
       		convertMatToEigen_Image(camera_frame_normalized, frame_eigen);
        	frame_eigen.resize(frame_eigen.rows()*frame_eigen.cols(),1);
        	for(int j=0;j<frame_eigen.rows();j++) sample_frames(i%coresetSampleRate,j) = frame_eigen(j,0);
        		cout<<"Mean image intensity for frame "<<i<<" is "<<frame_eigen.mean()<<endl;
        	YTrain.push_back(classID);
        }
	}
}

void serialPipelineKinect(int treeHeight, int features, int coresetSampleRate, float ratio){

	cv::Mat depthMat(Size(640,480), CV_16UC1);
	cv::Mat rgbMat(Size(640,480), CV_16UC3);

	Freenect::Freenect freenect;
	MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

	#ifdef SHOWFEED
		namedWindow("rgb",CV_WINDOW_AUTOSIZE);
		namedWindow("depth",CV_WINDOW_AUTOSIZE);
	#endif
	
	device.startVideo();
	device.startDepth();

	Eigen::MatrixXf sample_frames;
	float img_x = 480*ratio;
	float img_y = 640*ratio;
	sample_frames.resize(coresetSampleRate,img_x*img_y*3*2);
	std::vector<int> YTrain;
	std::vector<int> changeLoc;
	changeLoc.push_back(0);
	bool idChange=false;
	int classID = 0;
	bool isBreak=false;
	for(int i=0;i<500;i++){
		device.getVideo(rgbMat);
		device.getDepth(depthMat);
		//Warmup
	}
	for(int i=0;;i++){
		if(changeLoc.size()==0) changeLoc.push_back(0);
		
		device.getVideo(rgbMat);
		device.getDepth(depthMat);

		cv::Mat rgb_float,rgbf_resized,rgbf_normalized;
		rgbMat.convertTo(rgb_float,CV_32FC3);
		cv::resize(rgb_float,rgbf_resized,Size(img_y,img_x),0,0,INTER_LINEAR);
        cv::normalize(rgbf_resized,rgbf_normalized,0,1,NORM_MINMAX,CV_32FC3);

		cv::Mat depth_float,depthf_resized,depthf_normalized;
		depthMat.convertTo(depth_float,CV_32FC1);
		cv::resize(depth_float,depthf_resized,Size(img_y,img_x),0,0,INTER_LINEAR);
        cv::normalize(depthf_resized,depthf_normalized,0,1,NORM_MINMAX,CV_32FC3);


        #ifdef SHOWFEED
        	cv::Mat feed_image;
        	rgbf_normalized.copyTo(feed_image);
        	ostringstream classNum;
        	classNum << changeLoc.size();
        	std::string camera_text = "Learning class number: " + classNum.str();
        	cv::putText(feed_image, camera_text, cvPoint(10,10), FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(200,200,250));
        	cv::imshow("rgb",feed_image);
        	cv::imshow("depth",depthf_normalized);
        #endif
        
        getKeyStatus(classID,isBreak,idChange);
        if(idChange){
        	idChange=false;
        	changeLoc.push_back(i%coresetSampleRate);
        }
        if(isBreak) break;

        if(!(i%coresetSampleRate)&i>0){
			device.stopVideo();
			device.stopDepth();
        	std::cout<<"Sampling for coreset construction #"<<floor(i/coresetSampleRate)<<"..."<<std::endl;
        	Eigen::MatrixXf XTrain = sample_frames;

        	//std::string ROut = XTrainFile + ".model";

        	int nClasses = changeLoc.size();
        	std::cout<<"Number of training classes: "<<nClasses<<endl;
        	changeLoc.push_back(coresetSampleRate-1);
        	cv::Mat XTrain_Mat;
        	//convertEigenToMat(XTrain,XTrain_Mat);

        	std::vector<Eigen::MatrixXf> Coresets;
        	for(int ii=1;ii<changeLoc.size();ii++){
        		std::cout<<"Coreset details "<<changeLoc.at(ii-1)<<" "<<changeLoc.at(ii)<<" "<<XTrain.cols()<<endl;
        		Eigen::MatrixXf XTrain_cut1 = XTrain.block(changeLoc.at(ii-1),0,changeLoc.at(ii)-changeLoc.at(ii-1),XTrain.cols());
        		Eigen::MatrixXf XTrain_cut = XTrain_cut1;
        		//randomPermuteRows(XTrain_cut);
        		Eigen::MatrixXf Ctemp = pComputeCoresetTree(XTrain_cut,treeHeight,1);
        		Coresets.push_back(Ctemp);
        	}
        	sample_frames.resize(0,0);																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																														
        	//cv::normalize(disp_mat,disp_mat_normalized,0,1,NORM_MINMAX,CV_32FC3);
        	for(int cd=0;cd<Coresets.size();cd++){
        		for(int cc=0;cc<Coresets.at(cd).rows();cc++){
        			Eigen::MatrixXf Vec_i = Coresets.at(cd).row(cc);
        			Eigen::MatrixXf RGB_i = Vec_i.block(0,0,1,img_x*img_y*3);
        			Eigen::MatrixXf Depth_i = Vec_i.block(0,img_x*img_y*3,1,img_x*img_y);
        			RGB_i.resize(img_x,img_y*3);
        			Depth_i.resize(img_x,img_y);

        			cv::Mat RGBMat, DepthMat;
        			convertEigenToMat_Image(RGB_i,RGBMat);
        			EigenToMat(Depth_i,DepthMat);

        			ostringstream coresetnum;
        			coresetnum << cd;
        			coresetnum << " ";
        			coresetnum << cc;
        			std::string cctext = "Coreset RGB: " + coresetnum.str();
        			std::string cctext2 = "Coreset Depth: " + coresetnum.str();
        			cv::namedWindow(cctext,WINDOW_AUTOSIZE);
        			cv::imshow(cctext,RGBMat);
        			cv::namedWindow(cctext2,WINDOW_AUTOSIZE);
        			cv::imshow(cctext2,DepthMat);
        		}
        	}
        	cv::waitKey(0);
        	//We are now displaying the top image in the coreset. 


			std::cout<<"Constructed coresets..."<<endl;
			std::vector<std::vector<std::vector<double> > > ft_mat;
        	std::vector<cv::Mat> CoresetsMat;
        	for(int ev=0;ev<Coresets.size();ev++){
        		cv::Mat temp;
        		EigenToMat(Coresets.at(ev),temp);
        		CoresetsMat.push_back(temp);
        	}
			for(int Idx=0;Idx<Coresets.size();Idx++){
				std::vector<std::vector<double> > t;	
				getFeatures(features, CoresetsMat.at(Idx), t);
				ft_mat.push_back(t);
			}
			CoresetsMat.erase(CoresetsMat.begin(),CoresetsMat.end());
	

			std::vector<int> labels = getLabelMatrix(ft_mat,classID);			
			std::vector<std::vector<double> > ft_mat_lin;
			linearizeFeatureVector(ft_mat,ft_mat_lin);
			PModel ft_model = PTrain(0, ft_mat_lin, labels);
			ft_mat.erase(ft_mat.begin(),ft_mat.end());
			//PClassifier::writeModel(ft_model,ROut);

        	///Executing clearing commands.
        	YTrain.erase(YTrain.begin(),YTrain.end());
        	changeLoc.erase(changeLoc.begin(),changeLoc.end());
			sample_frames.resize(coresetSampleRate,img_x*img_y*3);
			device.startVideo();
			device.startDepth();
        }

        else{
       		Eigen::MatrixXf frame_eigen;
       		convertMatToEigen_ImageDepth(rgbf_normalized,depthf_normalized, frame_eigen);
       		std::cout<<frame_eigen.rows()<<" "<<frame_eigen.cols()<<std::endl;
        	frame_eigen.resize(frame_eigen.rows()*frame_eigen.cols(),1);
        	for(int j=0;j<frame_eigen.rows();j++) sample_frames(i%coresetSampleRate,j) = frame_eigen(j,0);
        		cout<<"Mean image intensity for frame "<<i<<" is "<<frame_eigen.mean()<<endl;
        	YTrain.push_back(classID);
        }
	}
	device.stopVideo();
	device.stopDepth();
}

class threadPipeline{
public:	
	boost::mutex coresetmtx_, featuremtx_, classifiermtx_, framemtx_;
	bool train;
	bool canComputeCoreset;
	std::vector<Eigen::MatrixXf> CoresetsList;
	std::vector<Eigen::MatrixXf> frames;
	std::vector<int> labels;
	std::vector<std::vector<double> > features;
	std::vector<int> vlabels;
	PModel model;
	int treeHeight;
	int featurescode;
	int coresetSampleRate;

	int classID;

	threadPipeline(float img_x, float img_y, int c){
		train=true;
		classID=0;
		coresetSampleRate=c;
		canComputeCoreset=true;
	}

	void thread_computeCoreset(){
		// This function removes coresetSampleRate number of points from the point matrix and computes their coreset.
		// It then appends this to the existing coreset list.
		framemtx_.lock();

		Eigen::MatrixXf newPoints, newLabels;
		newPoints.resize(coresetSampleRate,frames.at(0).cols());
		newLabels.resize(coresetSampleRate,1);
		for(int i=0;i<coresetSampleRate;i++){
			newPoints.row(i) = frames.at(i);
			newLabels(i,0) = labels.at(i);
		} 
		frames.erase(frames.begin(),frames.begin()+coresetSampleRate);
		labels.erase(labels.begin(),labels.begin()+coresetSampleRate);
		framemtx_.unlock();
		canComputeCoreset=true;
		//Cropping out the top part of the matrix and resizing it.

		std::vector<int> splits;
		splits.push_back(0);
		//Adding index zero (for the first split).

		int old_label=newLabels(0,0);
		for(int i=1;i<newLabels.rows();i++){
			int label = newLabels(i,0);
			if(label!=old_label) splits.push_back(i);
			old_label = label;
		}
		splits.push_back(newLabels.rows()-1);
		cout<<newLabels.rows()<<" "<<newPoints.rows()<<endl;
		//Adding index n (for the last split).
		for(int i=1;i<splits.size();i++){
			int label = newLabels(splits.at(i-1),0);
			Eigen::MatrixXf lframes = newPoints.block(splits.at(i-1),0,splits.at(i)-splits.at(i-1),newPoints.cols());
			Eigen::MatrixXf cframes = pComputeCoresetTree(lframes, treeHeight, 1);
			//Using RedSVD for SVD in coreset computation.
			coresetmtx_.lock();
			while(CoresetsList.size()<label+1) CoresetsList.push_back(MatrixXf::Zero(0,lframes.cols()));
			//If the class coreset is not present, add it.
			CoresetsList.at(label).conservativeResize(CoresetsList.at(label).rows()+cframes.rows(),lframes.cols());
			CoresetsList.at(label).block(CoresetsList.at(label).rows()-cframes.rows(),0,cframes.rows(),cframes.cols()) = cframes;
			coresetmtx_.unlock();	
			//Appending new coresets to existing coreset list.
		}	
		std::cout<<"Coreset computation completed, rejoining main."<<std::endl;
	}

	void thread_computeFeatures(Eigen::MatrixXf prefeat, int this_label){
		//This function spawns a thread which computes the features for every set of coresets (points).
		cv::Mat prefeatmat;
		EigenToMat(prefeat,prefeatmat);
		std::vector<std::vector<double> > t;
		getFeatures(featurescode, prefeatmat, t);
		featuremtx_.lock();
		for(int i=0;i<t.size();i++){
			vlabels.push_back(this_label);
			features.push_back(t.at(i));
		}
		featuremtx_.unlock();
		std::cout<<"Features computation completed, rejoining main."<<std::endl;
	}

	void thread_computeClassifier(int type){
		//This function spawns a thread which will learn a classifier.
		classifiermtx_.lock();
		model = PTrain(type,features,vlabels);
		classifiermtx_.unlock();
		std::cout<<"Classifier updated, rejoining main."<<std::endl;
	}

	void thread_addFrame(Eigen::MatrixXf frame_eigen){
		    frame_eigen.resize(1,frame_eigen.rows()*frame_eigen.cols());
		    framemtx_.lock();
		    frames.push_back(frame_eigen);
		    labels.push_back(classID);
       		framemtx_.unlock();
       		//std::cout<<"Frame added with mean value "<<frame_eigen.mean()<<std::endl;
	}

	void getKeyStatus(bool isBreak){
		int keyID = cv::waitKey(30);
		switch (keyID) {
			case 32:
				classID++;
				break;
			case 27:
				isBreak=true;
				break;
		}
    }
};

void threadedPipelineKinect(int treeHeight, int featurescode, int coresetSampleRate, float ratio){
	cv::Mat depthMat(Size(640,480), CV_16UC1);
	cv::Mat rgbMat(Size(640,480), CV_16UC3);

	Freenect::Freenect freenect;
	MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
	bool isBreak(false);


	#ifdef SHOWFEED
		namedWindow("rgb",CV_WINDOW_AUTOSIZE);
		namedWindow("depth",CV_WINDOW_AUTOSIZE);
	#endif
	
	device.startVideo();
	device.startDepth();

	float img_x = 480*ratio;
	float img_y = 640*ratio;

	threadPipeline T(img_x,img_y,coresetSampleRate);
	T.treeHeight=treeHeight;
	T.featurescode=featurescode;

	for(int i=0;i<10;i++){
		device.getVideo(rgbMat);
		device.getDepth(depthMat);
		//Warmup
	}

	clock_t start;
	while(true){
		start=clock();
		T.getKeyStatus(isBreak);

        if(isBreak) break;
        //We get the correct mode from the user before training/testing.

		if(T.train){//select if training mode is set.
			//Normal addition of image (and displaying it (?))

			if(T.frames.size()>coresetSampleRate & T.canComputeCoreset){//If there are more images than the sampling rate (sampling required).
				T.canComputeCoreset=false;
				boost::thread coresetThread(boost::bind(&threadPipeline::thread_computeCoreset, &T));
				coresetThread.detach();
			}

			for(int i=0;i<T.CoresetsList.size();i++){
				//If any of the classes have at least FEATURE_THRESH #points, then we will calculate their features.
				if(T.CoresetsList.at(i).rows()>FEATURE_THRESH){
					Eigen::MatrixXf prefeat = T.CoresetsList.at(i).block(T.CoresetsList.at(i).rows()-FEATURE_THRESH,0,FEATURE_THRESH,T.CoresetsList.at(i).cols());
					T.CoresetsList.at(i).conservativeResize(T.CoresetsList.at(i).rows()-FEATURE_THRESH,T.CoresetsList.at(i).cols());
					//Extracting the coresets and resizing the original matrix.

					boost::thread featuresThread(boost::bind(&threadPipeline::thread_computeFeatures,&T, prefeat, i));
					featuresThread.detach();
				}
			}

			if(!(T.features.size()%LEARNER_THRESH)&T.features.size()>0){
				//If significant number of features have been accumulated, we will train the learner.
				//No points will be removed however.
				boost::thread classifierThread(boost::bind(&threadPipeline::thread_computeClassifier,&T, 0));
			}
			device.getVideo(rgbMat);
			device.getDepth(depthMat);

			cv::Mat rgb_float,rgbf_resized,rgbf_normalized;
			rgbMat.convertTo(rgb_float,CV_32FC3);
			cv::resize(rgb_float,rgbf_resized,Size(img_y,img_x),0,0,INTER_LINEAR);
        	cv::normalize(rgbf_resized,rgbf_normalized,0,1,NORM_MINMAX,CV_32FC3);

			cv::Mat depth_float,depthf_resized,depthf_normalized;
			depthMat.convertTo(depth_float,CV_32FC1);
			cv::resize(depth_float,depthf_resized,Size(img_y,img_x),0,0,INTER_LINEAR);
        	cv::normalize(depthf_resized,depthf_normalized,0,1,NORM_MINMAX,CV_32FC3);


        	#ifdef SHOWFEED
        		cv::Mat feed_image;
        		rgbf_normalized.copyTo(feed_image);
        		ostringstream classNum;
        		classNum << T.classID;
        		std::string camera_text = "Learning class number: " + classNum.str();
        		cv::putText(feed_image, camera_text, cvPoint(10,10), FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(200,200,250));
        		cv::imshow("rgb",feed_image);
        		cv::imshow("depth",depthf_normalized);
        	#endif
        
        	Eigen::MatrixXf frame_eigen;
       		convertMatToEigen_ImageDepth(rgbf_normalized,depthf_normalized, frame_eigen);
       		boost::thread frameThread(boost::bind(&threadPipeline::thread_addFrame,&T,frame_eigen));
       		frameThread.detach();
       		//<< (std::clock()-start)/(double)CLOCKS_PER_SEC<<endl;
		}
		else{

		}
	}

}