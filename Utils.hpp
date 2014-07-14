 /*
 * Nikhil Naik
 * File IO - adopted from REDSVD library
 */ 

#ifndef UTILS_HPP  
#define UTILS_HPP  

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <algorithm>
#include <assert.h>
#include <iterator>
typedef std::vector<std::pair<int, float> > fv_t;

using namespace std;
using namespace Eigen;

float matrixL2norm(const MatrixXf& );

void setFooter();
void writeMatrix(const string& , const MatrixXf& );
void writeVector(const string& , const VectorXf& );
void readLine(const string& , fv_t& );
void readMatrix(const string& , MatrixXf& );
void readOneClassSparseData(const string& , SparseMatrix<float>& , int);
void readOneClassData(const string& , const string& ,SparseMatrix<float>& , int);	
vector<int> getNumClasses(const string&);
#endif 
