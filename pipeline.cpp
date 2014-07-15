#include "PBoostPipeline.hpp"
#include "Utils.hpp"


int main(int argc,char* argv[]){
	switch (atoi(argv[1])){
		case 0:
		serialPipelineKinect(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atof(argv[5]));
		break;
		case 1:
		threadedPipelineKinect(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atof(argv[5]));
		break;
	}
	return 0;
}
