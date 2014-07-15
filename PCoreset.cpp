#include "PCoreset.hpp"

void pWriteSparseMatrix(const string fn, const MatrixXf& M){
	/* This function will write the coreset to the specified libSVM format. 
	LibSVM follows a sparse format, where zeros are not to be stored. For data:
	1 2 0 4 2
	We will store it as
	1:1 2:2 4:4 5:2 */
	ofstream outfile (fn.c_str());
	if(outfile.is_open()){
  		for (int i = 0; i < M.rows(); ++i){
    			for (int j = 0; j < M.cols(); ++j){
				if(M(i,j)!=0) //fprintf(outfp, "%d:%f ", j+1, M(i,j));
				outfile << j+1 << ":" << M(i,j) << " ";
    			}
    			outfile << endl;
  		}
	}
	outfile.close();
  	//fclose(outfp);
  	cout << "wrote " << fn << endl; 
}

void pReadSparseData(const string fn, vector<SparseMatrix<float> >& A){
	/* Read a sparse dataset (LibSVM-type) parallelly loading examples of each class. 
	TODO:This method might take up a lot of memory initially, need to fix that.(Probably by splitting the dataset?)*/
	vector<int> classes = getNumClasses(fn);
	omp_set_num_threads(classes.size());
	for(int i=0;i<classes.size();i++){
		SparseMatrix<float> temp;
		A.push_back(temp);
	}
	
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		readOneClassSparseData(fn, A.at(id),classes.at(id));
	}
}

void pReadData(const string fn,const string fn2, vector<SparseMatrix<float> >& A){
	/* Read a normal dataset parallelly loading rows of each class. */
        vector<int> classes = getNumClasses(fn2);
        omp_set_num_threads(classes.size());
        for(int i=0;i<classes.size();i++){
                SparseMatrix<float> temp;
                A.push_back(temp);
        }

        #pragma omp parallel
        {
                int id = omp_get_thread_num();
                readOneClassData(fn, fn2, A.at(id),classes.at(id));
        }
}

void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove){
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

void removeColumn(Eigen::MatrixXf& matrix, unsigned int colToRemove){
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void randomPermuteRows(MatrixXf input1, MatrixXf input2, MatrixXf& output){
	// This function randomly merges the rows of the two input matrices, although the within matrix row order is maintained.
	std::vector<Eigen::MatrixXf> input;
	input.push_back(input1);
	input.push_back(input2);
	int totalrows = input1.rows()+input2.rows();
	for(int i=0;i<totalrows;i++){
		int choice = rand()>0.5? 0 : 1;
		if(input.at(choice).rows()==0) choice = (choice+1)%2;
		output.row(i) = input.at(choice).row(0);
		removeRow(input.at(choice),0);
	}
}

MatrixXf pComputeCoresetTree(MatrixXf input, int treeHeight, int svdType, int numCoresetPoints, int extraResizePoints){
	/* This function constructs the coreset tree. Input parameters:
		input 				: Matrix input data for coreset construction.
		treeHeight 			: Height of tree to be constructed from the input data.
		svdType 			: Type of SVD to be used to compute the coreset at each node. 
									0  - Standard Eigen SVD
									1  - RedSVD
		numCoresetPoints	: Optional parameter, for number of rows in each coreset. Default value is #input-rows/#leaf-nodes.
		extraResizePoints	: Optional parameter, number of rows in final coreset, obtained after an additional SVD round.
	*/

	if(!numCoresetPoints) numCoresetPoints = input.rows()/(pow(2.0,treeHeight-1));

	std::vector<Eigen::MatrixXf> splitInput(pow(2.0,treeHeight-1));
	//Input matrix for levels > 1 of the tree formed. Level 1 will be formed directly by splitting the input.

	std::vector<Coreset> output(pow(2.0,treeHeight)-1);
	//Output coreset vector. Final output will be the root of the tree, hence size(output) after all iterations will be 1.

	int nodeSum = 0;
	// Number of nodes already traversed in the tree.(DEBUG PARAMETER)
	int nSVD = 0;
	// Number of total SVDs carried out. (DEBUG PARAMETER)

	for(int i=0;i<treeHeight;i++){
		int nNodes = pow(2.0,treeHeight-1-i);
		omp_set_num_threads(nNodes);
		cout<<"Setting #threads as "<<nNodes<<endl;
		//For each level in the tree, we find out the #nodes, and the coreset computation for each level is done in parallel.

		#pragma omp parallel
		{
			int j = omp_get_thread_num();
			Coreset temp;
			temp.setPoints(numCoresetPoints);
			if(i==0) temp.setMatrix(input.block(j*numCoresetPoints,0,numCoresetPoints,input.cols()));
			else temp.setMatrix(splitInput.at(j));
			output.at(j) = temp;
			output.at(j).computeCoreset(svdType);
			//Coreset computed.
				
			if(extraResizePoints&i!=0){
				output.at(j).setPoints(extraResizePoints);
				output.at(j).computeCoreset(svdType);
			}
			//Additional resize with an SVD Round.
		}
		nSVD+= extraResizePoints&i!=0? 2*nNodes : nNodes;

		//The following part modifies splitInput to take the value of the previously computed coresets.
		if(nNodes!=1){
			splitInput.resize(nNodes/2);

			#ifdef PTREE_DEBUG
				cout<<"resize successful"<<endl;
			#endif

			for(int j=0;j<splitInput.size();j++){
				/// Merging two coresets to form one. Currently just concatenating rows. ////
				Eigen::MatrixXf temp1 = output.at(2*j).getMatrix();
				Eigen::MatrixXf temp2 = output.at(2*j+1).getMatrix();
				Eigen::MatrixXf temp(output.at(2*j).getMatrix().rows()+output.at(2*j+1).getMatrix().rows(),input.cols());

				//randomPermuteRows(temp1,temp2,temp);
				splitInput.at(j) = temp;
				splitInput.at(j) << output.at(2*j).getMatrix(),
									output.at(2*j+1).getMatrix();
				#ifdef PTREE_DEBUG
					cout<<"Added coreset at "<<nodeSum+nNodes+j<<" by merging "<<nodeSum+2*j<<" and "<<nodeSum+2*j+1<<endl;
				#endif
				/// End merge.															 ////
			}
			output.resize(nNodes/2);
		}
		nodeSum+=nNodes;	
	}
	cout << "Total number of SVDs done is:" << nSVD << endl;
	return output.at(0).getMatrix();
	//Return the coreset matrix for the root node.
}

