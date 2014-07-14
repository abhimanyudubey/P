#include "CoresetStack.hpp"
#include "Utils.hpp"
#include "omp.h"
#define CORESETSTACK_DEBUG 1
#define CORESETDEBUG 1
using namespace std;

int main(void)
{
	int N = 20; 
	int d = 10;		
	int npoints = 6;
	int svd_method = 0;
	Eigen::MatrixXf A = Eigen::MatrixXf::Random(N,d);	
	writeMatrix("file2", A);	
	Eigen::MatrixXf A1; readMatrix("file2", A1);	
	//Eigen::MatrixXf A1 = Eigen::MatrixXf::Random(N,d);	
	Eigen::MatrixXf A2 = Eigen::MatrixXf::Random(N,d);	
	Eigen::MatrixXf A3 = Eigen::MatrixXf::Random(N,d);	
	Eigen::MatrixXf A4 = Eigen::MatrixXf::Random(N,d);
	Eigen::MatrixXf A5 = Eigen::MatrixXf::Random(N,d);	
	Eigen::MatrixXf A6 = Eigen::MatrixXf::Random(N,d);
	Eigen::MatrixXf A7 = Eigen::MatrixXf::Random(N,d);
	Eigen::MatrixXf A8 = Eigen::MatrixXf::Random(N,d);	
	
	Eigen::MatrixXf C;
	CoresetStack cor_stack(svd_method);
	Coreset c1(A1, npoints);
	Coreset c2(A2, npoints);	
	Coreset c3(A3, npoints);
	Coreset c4(A4, npoints);
	Coreset c5(A2, npoints);
	Coreset c6(A2, npoints);	
	Coreset c7(A3, npoints);
	Coreset c8(A4, npoints);
		
	cout << "c1" << endl;	
	cor_stack.push(c1);	
	cout << "c2" << endl;		
	cor_stack.push(c2);
	cout << "c3" << endl;		
	cor_stack.push(c3);
	cout << "c4" << endl;		
	cor_stack.push(c4);	
	cout << "c5" << endl;	
	cor_stack.push(c5);	
	cout << "c6" << endl;		
	cor_stack.push(c6);
	cout << "c7" << endl;		
	cor_stack.push(c7);
	cout << "c8" << endl;		
	cor_stack.push(c8);	
	
	Coreset c9 = cor_stack.top();
	cout << "level : "  << c9.getLevel() << endl;
	cout << "c12345678 : "  << endl << c9.getMatrix() << endl;
		
    return(0);
}
