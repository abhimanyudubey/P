#include "StaticData.hpp"
#include "Utils.hpp"

using namespace std;

int main(void)
{
	int N = 1000; 
	int d = 100;		
	int npoints = 200;
	int svd_method = 0;
	float eps = 0.1;
	float j = ceil(float(npoints + 1)/(1 + 1.0/eps));
	j = 2;
	npoints = j + ceil(j/eps) - 1;
	//Eigen::MatrixXf A = Eigen::MatrixXf::Random(N,d);
	Eigen::MatrixXf A; readMatrix("Atest", A);		
	Eigen::MatrixXf C1, C2;
	
	Coreset c1(A, npoints);
	c1.computeCoresetSV(0);
	C1 = c1.getMatrix();
	RedSVD svd0(A, npoints, 0);
	
	Coreset c2(A, npoints);
	c2.computeCoresetSV(1);
	C2 = c2.getMatrix();
	RedSVD svd1(A, npoints, 1);
	
	/*	
	cout << "c1" << endl;		
	cout << C1 << endl;
	
	cout << "c2" << endl;		
	cout << C2 << endl;
	*/
	
	Eigen::MatrixXf Q = Eigen::MatrixXf::Identity(d,d-j);
	
	Eigen::MatrixXf costA = A*Q;
	Eigen::MatrixXf costC1 = C1*Q;
	Eigen::MatrixXf costC2 = C2*Q;
	
	float cost_A = costA.squaredNorm();
	float cost_c1 = costC1.squaredNorm() + svd0.SingularNorm();
	float cost_c2 = costC2.squaredNorm() + svd1.SingularNorm();
	cout << cost_A << "\t" << cost_c1 << "\t" << cost_c2 << endl;
	cout << svd0.SingularNorm() << "\t" << svd1.SingularNorm() << "\t"  << endl;
	
	cout << " Fro error norm for JacobiSVD  " << abs(cost_c1/cost_A - 1) << endl; 		
	cout << "Fro error norm for RedSVD  " << abs(cost_c2/cost_A - 1) << endl; 		
	
	cout << matrixL2norm(costC1) << endl << endl;
	cout  << matrixL2norm(costC2) << endl << endl;
		
	cout << " L2 error norm for JacobiSVD  " << abs(pow(matrixL2norm(costC1),2)/pow(matrixL2norm(costA),2) - 1) << endl; 		
	cout << "L2 error norm for RedSVD  " << abs(pow(matrixL2norm(costC2),2)/pow(matrixL2norm(costA),2) - 1) << endl; 		
	
	/*		
	CoresetStack cor_stack;
	C2 = computeCoresetTree(A, npoints, npoints, svd_method);	
	cout << "c2" << endl;		
	cout << C2 << endl;
	*/
			
    return(0);
}
