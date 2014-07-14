#include "StaticData.hpp"
#include "Utils.hpp"
#include <ctime>

using namespace std;

int main(int argc, char* argv[])
{
	if (argc == 1){
		setFooter();
		return 0;
	  }
	
	clock_t t0=clock(),t1;
	string fname = argv[1];
	Eigen::MatrixXf A; 
	readMatrix(argv[1], A);		 
	cout << "Read Matrix "<< argv[1] << " with: "  << A.rows() 
						<< " datapoints and " << A.cols() << " features"<< endl;			
	MatrixXf C  = computeCoresetTree(A, atoi(argv[2]), atoi(argv[2]), atoi(argv[3]));	
	cout << "Constructed coreset with " << C.rows() 
			<< " datapoints" << endl; 	

	cout << "Time taken in seconds  " << (double)(clock() - t0)/CLOCKS_PER_SEC << endl;
    string npts(argv[2]);
    string cname = "coreset_" + npts + "_" + fname;  
    writeMatrix(cname, C);
	
	/*
	t1 =  clock();		
	
	Eigen::MatrixXf B;readMatrix("a9a_Xpos", B);		
	MatrixXf C2  = computeCoresetTree(B, npoints, npoints, svd_method);	
	writeMatrix("cor1_a9a_Xpos", C2);			
	cout << "Time taken in seconds  " << (double)(clock() - t1)/CLOCKS_PER_SEC << endl;
	*/	
		
    return(0);
}
