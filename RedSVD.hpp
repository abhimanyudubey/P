#ifndef REDSVD_HPP  
#define REDSVD_HPP  

#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace Eigen;

class RedSVD 
{
	friend class Coreset;
	 
public:   
 
	RedSVD(const MatrixXf& ,const int ,const int );
	void sampleTwoGaussian(float& , float& );
	void sampleGaussianMat(MatrixXf& );
	void processGramSchmidt(MatrixXf& );
	float SingularNorm();	 
private:

	Eigen::MatrixXf matU_;
	Eigen::MatrixXf matS_;
	Eigen::MatrixXf matV_;
};

#endif
