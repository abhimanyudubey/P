/*
 * Nikhil Naik
 */ 

#ifndef CORESET_HPP  
#define CORESET_HPP  

//#define CORESET_DEBUG

//put just the descriptive part of the class in here 
#include <vector>
#include <iostream>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "RedSVD.hpp"

class Coreset 
{ 
public:   
  //static const int npoints = 4;
  Coreset();
  Coreset(Eigen::MatrixXf ,int );
  Coreset(int );
  void setMatrix(Eigen::MatrixXf );
  Eigen::MatrixXf getMatrix(); 
  void setLevel(int );
  void setPoints(int );
  int getLevel();  
  int rows();
  int cols();
  void computeCoreset(const int ); 
  void computeCoresetSV(const int );
//  Coreset mergeCoreset(Coreset );
  void mergeCoreset(Coreset& ,const int );

private: 
   Eigen::MatrixXf matCor_;
   int level;
   int npoints;
}; 
#endif 
