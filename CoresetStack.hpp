 /*
 * Nikhil Naik
 */ 

#ifndef CORESETSTACK_HPP  
#define CORESETSTACK_HPP  

//#define CORESETSTACK_DEBUG

#include "Coreset.hpp"
#include "RedSVD.hpp"
#include <stack>

using namespace Eigen;

class CoresetStack
{
//friend class StaticData;	 
public:   
	CoresetStack(int );
	void push(Coreset );
	Coreset top();
	void pop();
	int size();
	int getSVDmethod();
private: 
	std::stack<Coreset> core_stack;
	int svd_method;
}; 
#endif 
