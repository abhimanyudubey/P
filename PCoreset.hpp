
#ifndef PCORESET_HPP
#define PCORESET_HPP

#include "CoresetStack.hpp"
#include "Utils.hpp"
#include "omp.h"
#include <ctime>
#include <math.h>
#include <vector>

using namespace std;
using namespace Eigen;

void pWriteSparseMatrix(const string, const MatrixXf&);
void pReadSparseData(const string, vector<SparseMatrix<float> >&);
void pReadData(const string, const string, vector<SparseMatrix<float> >&);
void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove);
void removeColumn(Eigen::MatrixXf& matrix, unsigned int colToRemove);
void randomPermuteRows(MatrixXf input1, MatrixXf input2, MatrixXf& output);
MatrixXf pComputeCoresetTree(MatrixXf, int, int svdType = 0, int numCoresetPoints = 0, int extraResizePoints = 0);
#endif

