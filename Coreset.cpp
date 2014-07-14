#include "Coreset.hpp"  
#define CORESET_DEBUG 1
//const int Coreset::npoints;

Coreset::Coreset()
{
	level = 0;
}

Coreset::Coreset(Eigen::MatrixXf cor, int n_coreset_points)
{
	matCor_ = cor;
	level = 0;
	npoints = n_coreset_points;
}

Coreset::Coreset(int l)
{
	level = l;
}

void Coreset::setMatrix(Eigen::MatrixXf input) 
{ 
  matCor_ = input; 
} 

void Coreset::setPoints(int n_points){
	npoints=n_points;
}

Eigen::MatrixXf Coreset::getMatrix() 
{ 
  return matCor_; 
} 

void Coreset::setLevel(int l) 
{ 
  level = l; 
} 

int Coreset::getLevel() 
{ 
  return level; 
} 

int Coreset::rows() 
{ 
  return matCor_.rows(); 
} 

int Coreset::cols() 
{ 
  return matCor_.cols(); 
} 
 
void Coreset::mergeCoreset(Coreset& c2, const int method)
{
	//Coreset c3;
	Eigen::MatrixXf m1 = getMatrix();
	Eigen::MatrixXf m2 = c2.getMatrix();
	Eigen::MatrixXf m3(m1.rows() + m2.rows(), m1.cols());	 
 	m3 << m1,    
		 m2; // concatenate vertically
    #ifdef CORESET_DEBUG	
	//	std:: cout  << "rows_merge" << m3.rows() << std::endl;	 
	 #endif
			 
	setMatrix(m3);	
	computeCoreset(method);	
	level++;
	//delete &c2; 	
}             
    
void Coreset::computeCoreset(const int svd_method) 
{ 
  const Eigen::MatrixXf input = getMatrix();
  
  RedSVD svd(input, npoints, svd_method);
  /* 
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(input, 
							Eigen::ComputeThinU | Eigen::ComputeThinV);
  
  const  Eigen::MatrixXf& u =  svd.matrixU(); 
  const  Eigen::VectorXf& svec =  svd.singularValues();
  const  Eigen::MatrixXf& s = svec.asDiagonal();
  Eigen::MatrixXf v =  svd.matrixV();	
  v.transposeInPlace();  
  */ 

	
  svd.matV_.transposeInPlace();
  
      #ifdef CORESET_DEBUG	
		std:: cout  << "input: " << input.rows()
		<< "\t" <<input.cols() << std::endl;	 	 
		std:: cout  << "u: " << svd.matU_.rows()  
		<< "\t" << svd.matU_.cols() << std::endl;	 
		std:: cout  << "s: " << svd.matS_.rows()  
		<< "\t" << svd.matS_.cols() << std::endl;	 
		std:: cout  << "v: " << svd.matV_.rows()  
		<< "\t" << svd.matV_.cols() << std::endl;	 		
	#endif
	
	Eigen::MatrixXf output;
	 if(svd_method == 0)
	 {
		 
		if(npoints > input.cols())
		{ 
		output =  
			
			svd.matU_.block(0,0,npoints,input.cols())* 
			svd.matS_.block(0,0,input.cols(),input.cols())* 
			svd.matV_.block(0,0,input.cols(),input.cols()); 	
		
		}
		else
		{
	    output =
			
			svd.matU_.block(0,0,npoints,npoints)* 
			svd.matS_.block(0,0,npoints,npoints)* 
			svd.matV_.block(0,0,npoints,input.cols()); 
		}	  			 
	 }
	 if(svd_method == 1)
	 {
		output =  
			svd.matU_.block(0,0,npoints,npoints)* 
			svd.matS_.block(0,0,npoints,npoints)* 
			svd.matV_.block(0,0,npoints,input.cols()); 				 
	 }
	 
    #ifdef CORESET_DEBUG	
		std:: cout  << "output: " << output.rows()
		<< "\t" <<output.cols() << std::endl;	 	 
	#endif
   
  setMatrix(output);   
}

void Coreset::computeCoresetSV(const int svd_method) 
{ 
  Eigen::MatrixXf input = getMatrix();
  
  RedSVD svd(input, npoints, svd_method);  
  svd.matV_.transposeInPlace(); 
    
  Eigen::MatrixXf output = 
	  svd.matS_.block(0,0,npoints,npoints)* 
	  svd.matV_.block(0,0,npoints,input.cols()); 	
   
  setMatrix(output);   
}
