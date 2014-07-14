 /*
 * Nikhil Naik
 * Collection of scripts for static data case
 */ 
#include "StaticData.hpp"

MatrixXf returnCoreset(CoresetStack cstack)
{	
	#ifdef STATICDATA_DEBUG 
		std::cout << "final stack size  " << cstack.size() << std::endl;	
	#endif
		
	Coreset c1, c2;
	while(cstack.size()>1)
		{
			c1 = cstack.top();
			#ifdef STATICDATA_DEBUG 
			 std::cout << "c1: " << c1.getLevel() << std::endl;
			#endif
			cstack.pop();
			c2 = cstack.top();
			#ifdef STATICDATA_DEBUG
			 std::cout << "c2: " << c2.getLevel() << std::endl;
			#endif
		
			cstack.pop();
			c2.mergeCoreset(c1, cstack.getSVDmethod());
			cstack.push(c2);
		}
	c1 = cstack.top();
	return c1.getMatrix();
}

MatrixXf returnCoresetTopLevel(CoresetStack cstack)
{	
	#ifdef STATICDATA_DEBUG 
		std::cout << "final stack size  " << cstack.size() << std::endl;	
	#endif
		
	Coreset c1, c2;
	while(cstack.size()>1)
		{
			cstack.pop();
		}
	c1 = cstack.top();
	return c1.getMatrix();
}

MatrixXf computeCoresetTree(MatrixXf input,
			int n_coreset_points, int n_total_points, int svd_method) 
{	
	int nrows = input.rows();
	int ncols = input.cols();
	int nCount = 0;
	CoresetStack cstack(svd_method);
	for(int i = nrows - n_coreset_points; 
			i > -1; i -= n_coreset_points)
	    {
		 #ifdef STATICDATA_DEBUG 
			//std::cout << "Pushing Coreset " << i << std::endl;
			nCount++;
		 #endif	
			
		 cstack.push(Coreset(
			input.block(i, 0, n_coreset_points, ncols),n_coreset_points));
		 // delete last n_coreset_points using resize fn
		 input.conservativeResize(i,ncols);   	 
		}
		 #ifdef STATICDATA_DEBUG 
			std::cout << "Number of Coresets " << nCount << std::endl;
		 #endif	
		
	//return returnCoreset(cstack); 	
	return returnCoresetTopLevel(cstack); 				
}


