#include "Utils.hpp"

void setFooter()
{
cout << "use the following format :" << endl
<< "mainStaticData input_file_name n_coreset_points svd_method"
<< endl << "svd_method = 0 - default SVD from Eigen Library" << endl
<< "svd_method = 1 - redsvd" << endl;	 
}

float matrixL2norm(const MatrixXf& M)
{
	JacobiSVD<MatrixXf> svd(M, ComputeThinU | ComputeThinV);
	const  Eigen::VectorXf& svec =  svd.singularValues();
	//cout << endl << "largest eigen value: " << svec(0) << endl;
	return svec(0);	
}


void writeMatrix(const string& fn, const MatrixXf& M){
  FILE* outfp = fopen(fn.c_str(), "wb");
  if (outfp == NULL){
    throw string("cannot open ") + fn;
  }

  for (int i = 0; i < M.rows(); ++i){
    for (int j = 0; j < M.cols(); ++j){
      fprintf(outfp, "%+f ",  M(i, j));
    }
    fprintf(outfp, "\n");
  }

  fclose(outfp);
  cout << "wrote " << fn << endl; 
}

void writeVector(const string& fn, const VectorXf& V){
  cout << "write " << fn << endl;
  FILE* outfp = fopen(fn.c_str(), "wb");
  if (outfp == NULL){
    throw string("cannot open ") + fn;
  }

  for (int i = 0; i < V.rows(); ++i){
    fprintf(outfp, "%+f\n", V(i));
  }

  fclose(outfp);
}

void readLine(const string& line, fv_t& fv)
{
  istringstream is(line);

  int id;
  char sep;
  float val;
  while (is >> id >> sep >> val){
    fv.push_back(make_pair(id, val));
  }
  sort(fv.begin(), fv.end());
  fv.erase(unique(fv.begin(), fv.end()), fv.end());
}

void readMatrix(const string& fn, MatrixXf& A)
{
  ifstream ifs(fn.c_str());
  if (!ifs){
    throw string("failed to open " ) + fn;
  }

  vector< vector<float> > vs;
  for (string line; getline(ifs, line); ){
    istringstream is(line);
    vector<float> v; 
    float val;
    while (is >> val){
      v.push_back(val);
    }
    vs.push_back(v);
  }

  size_t rowN = vs.size();
  if (rowN == 0) return;
  size_t colN = vs[0].size();
  A.resize(rowN, colN);
  
  for (size_t i = 0; i < rowN; ++i){
    if (colN != vs[i].size()){
      cerr << "warning: " << i+1 << "-th row has " 
		   << vs[i].size() << " entries. " 
	       << colN << " entries are expected" << endl;
    }
    size_t colNmin = min(colN, vs[i].size());
    for (size_t j = 0; j < colNmin; ++j){
      A(i, j) = vs[i][j];
    }
  }
}

void readOneClassSparseData(const string& fn, SparseMatrix<float>& A, int choice){
	/* Read data belonging to class i in the dataset pointed to by fn.
	   The dataset should be in the sparse format that is used by LibSVM. Rows of the data would be like:
		#class <feature-id1>:<feature-value1> <feature-id2>:<feature-value2> ...
	For example,
	
		-1 1:2 2:4 16:2.34 ...
		
	This data only contains non zero elements.
	*/
	
	ifstream ifs(fn.c_str());
  	if (!ifs){
    		throw string("failed to open " ) + fn;
  	}

	vector<vector<float> > vs;
	vector<vector<int> > indexs;
	for(string line; getline(ifs, line);){
		for(int i=0;i<line.length();i++){
			if(line[i]==':') line[i] = ' ';
		}
		istringstream is(line);
		vector<float> values;
		vector<int> indices;
		float ind;
		int val,classval;
		is >> classval;
		if(classval==choice){
			while(is >> val){
				indices.push_back(val);
				is >> ind;
				values.push_back(ind);
			}
		}		
	}
	
	int maxVal;
	for(int i=0;i<vs.size();i++){
		for(int j=0;j<vs[i].size();j++){
			int d = indexs.at(i).at(j);
			if(d>maxVal) maxVal=d;
		}
	}
	A.resize(vs.size(),maxVal);
	A.setZero();
	
	assert(vs.size()==indexs.size());
	for(int i=0;i<vs.size();i++){
		assert(vs[i].size()==indexs.size());
		for(int j=0;j<vs[i].size();j++){
			int ind = indexs.at(i).at(j);
			float val = vs.at(i).at(j);
			 A.coeffRef(i,ind) = val;
		}
	}
	cout<<"The read one class sparse Matrix has "<<A.rows()<<" rows and "<<A.cols()<<" columns."<<endl;
}

void readOneClassData(const string& fn, const string& fn2, SparseMatrix<float>& A, int choice){
	/* Read data belonging to class i in the dataset pointed to by fn with label set pointing to by fn2.
		For example, a call would be like:
		readOneClassData("data_x","data_y",A,1) will read data from data_x with labels at data_y, whenever the class number is 1.
	*/
	
	ifstream ifs(fn.c_str());
  	if (!ifs) throw string("failed to open " ) + fn;
	ifstream ifs2(fn2.c_str());
	if (!ifs2) throw string("failed to open ") + fn2;

	vector<vector<float> > vs;
	for(string line; getline(ifs, line);){
		istringstream is(line);
		string line2;
		istringstream is2(line2);
		vector<int> classes;
		int cindex;
		bool contains=false;
		while(is2 >> cindex){
			classes.push_back(cindex);
			if(cindex==choice) contains=true;		
		}
		vector<float> values;
		float ind;
		if(!contains){
			while(is >> ind){
				values.push_back(ind);
			}
		}		
	}
	
	A.resize(vs.size(),vs.at(0).size());
	A.setZero();
	
	for(int i=0;i<vs.size();i++){
		assert(vs[i].size()==vs[0].size());
		for(int j=0;j<vs[i].size();j++){
			A.coeffRef(i,j) = vs.at(i).at(j);
		}
	}
	cout<<"The read one class sparse Matrix has "<<A.rows()<<" rows and "<<A.cols()<<" columns."<<endl;
}
vector<int> getNumClasses(const string& fn){
	/* Returns the number of classes present in a multiclass dataset */
	ifstream ifs(fn.c_str());
	assert(ifs);
	vector<int> classes;
	for(string line; getline(ifs, line);){
		istringstream is(line);
		int classVal;
		is >> classVal;
		int i;
		for(i=0;i<classes.size();i++){
			if(classes.at(i)==classVal) break;
		}
		if(i==classes.size()) classes.push_back(classVal);
	}
	return classes;
}

