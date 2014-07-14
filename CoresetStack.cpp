#include "CoresetStack.hpp"  
#define CORESETSTACK_DEBUG 1
CoresetStack::CoresetStack(int m)
{
	svd_method = m;
}

int CoresetStack::getSVDmethod()
{
	return svd_method;
}

Coreset CoresetStack::top()
{
	if(!core_stack.empty())	
		return core_stack.top();
	else
		return Coreset(-1);
}

void CoresetStack::push(Coreset cor)
{
  Coreset c1 = cor;
  Coreset c2 = top();
  		#ifdef CORESETSTACK_DEBUG
  		 std::cout << "c2: " << c2.getLevel() << std::endl;
		 std::cout << "c1: " << c1.getLevel() << std::endl;					
		#endif
				
  while(c2.getLevel() ==  c1.getLevel())
	 {	
		 core_stack.pop();
		 c1.mergeCoreset(c2, svd_method);
		 
		 #ifdef CORESETSTACK_DEBUG		
		  std::cout << "coresets merged" << std::endl;
		 #endif
		  
		 c2 = top();	
		 
		 #ifdef CORESETSTACK_DEBUG		
		  std::cout << "l2: " << c2.getLevel() << std::endl;
		  std::cout << "l1: " << c1.getLevel() << std::endl;					
		 #endif
		
	 }
	 
 #ifdef CORESETSTACK_DEBUG 	 	 	 
	std::cout << "coresets pushed" << std::endl;		 	  	 
 #endif
 		
 core_stack.push(c1);			
}
	
void CoresetStack::pop()
{
		if(!core_stack.empty())	
			core_stack.pop();
}

int CoresetStack::size()
{
	return core_stack.size();
}












