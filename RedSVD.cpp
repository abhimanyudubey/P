/*
 * code adopted from the RedSVD library http://code.google.com/p/redsvd/
 */
#include "RedSVD.hpp"

const float SVD_EPS = 0.0001f;

RedSVD::RedSVD(const MatrixXf& A, const int rank, const int method)
{
	if (method == 0) // jacobi SVD
	{
	   Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, 
					Eigen::ComputeThinU | Eigen::ComputeThinV);
	  
	   matU_ =  svd.matrixU(); 
	   Eigen::VectorXf svec =  svd.singularValues();
	   matS_ = svec.asDiagonal();
	   matV_ = svd.matrixV();	
	}
	if (method == 1) // REDSVD
	{
		if (A.cols() == 0 || A.rows() == 0) return;
		 //int r = (rank < A.cols()) ? rank : A.cols();
		 //r = (r < A.rows()) ? r : A.rows();
		int r = rank;
		// Gaussian Random Matrix for A^T
		MatrixXf O(A.rows(), r);
		sampleGaussianMat(O);
		
		// Compute Sample Matrix of A^T
		MatrixXf Y = A.transpose() * O;
		
		// Orthonormalize Y
		processGramSchmidt(Y);

		// Range(B) = Range(A^T)
		MatrixXf B = A * Y;
		
		// Gaussian Random Matrix
		MatrixXf P(B.cols(), r);
		sampleGaussianMat(P);
		
		// Compute Sample Matrix of B
		MatrixXf Z = B * P;
		
		// Orthonormalize Z
		processGramSchmidt(Z);
		
		// Range(C) = Range(B)
		MatrixXf C = Z.transpose() * B; 
		
		JacobiSVD<MatrixXf> svdOfC(C,  ComputeThinU |  ComputeThinV);
			
		matU_ = Z * svdOfC.matrixU();
		Eigen::VectorXf svec =  svdOfC.singularValues();
	    matS_ = svec.asDiagonal();
		matV_ = Y * svdOfC.matrixV();			
	}
}

void RedSVD::sampleTwoGaussian(float& f1, float& f2){
  float v1 = (float)(rand() + 1.f) / ((float)RAND_MAX+2.f);
  float v2 = (float)(rand() + 1.f) / ((float)RAND_MAX+2.f);
  float len = sqrt(-2.f * log(v1));
  f1 = len * cos(2.f * M_PI * v2);
  f2 = len * sin(2.f * M_PI * v2);
}

void RedSVD::sampleGaussianMat(MatrixXf& mat){
  for (int i = 0; i < mat.rows(); ++i){
    int j = 0;
    for ( ; j+1 < mat.cols(); j += 2){
      float f1, f2;
      sampleTwoGaussian(f1, f2);
      mat(i,j  ) = f1;
      mat(i,j+1) = f2;
    }
    for (; j < mat.cols(); j ++){
      float f1, f2;
      sampleTwoGaussian(f1, f2);
      mat(i, j)  = f1;
    }
  }
} 

void RedSVD::processGramSchmidt(MatrixXf& mat){
  for (int i = 0; i < mat.cols(); ++i){
    for (int j = 0; j < i; ++j){
      float r = mat.col(i).dot(mat.col(j));
      mat.col(i) -= r * mat.col(j);
    }
    float norm = mat.col(i).norm();
    if (norm < SVD_EPS){
      for (int k = i; k < mat.cols(); ++k){
	mat.col(k).setZero();
      } 
      return;
    }
    mat.col(i) *= (1.f / norm);
  }
}

float RedSVD::SingularNorm()
{
		return matS_.squaredNorm();
}
