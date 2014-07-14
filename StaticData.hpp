#ifndef STATICDATA_HPP  
#define STATICDATA_HPP  

#define STATICDATA_DEBUG

#include "CoresetStack.hpp"

MatrixXf returnCoreset(CoresetStack );
MatrixXf returnCoresetTopLevel(CoresetStack );
MatrixXf computeCoresetTree(MatrixXf ,int , int ,int );

#endif 
