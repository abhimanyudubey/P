#!/bin/bash
#Training script for Parallel Coreset Construction
#Abhimanyu Dubey
#Camera Culture, MIT Media Lab
#June 2014

echo $1 $2 $3 $4 $5 $6
#Echoing all the variables for debugging
#Temporary setup:
#$1 - Dataset name.
#$2 - <CoresetTree Param>Height of tree required.
#$3 - <CoresetTree Param>#Rows in the leaf nodes.
#$4 - <CoresetTree Param>#Rows after coreset construction (additional SVDRound).(0 for no additional SVDRound)`
#$5 - Training method (default is LibSVM), will add others later on.
#$6 - LibSVM/Other classifier parameters.

#Setting default parameters
$5=${5-libsvm}
$6=${6-}

#Setting datapaths.

#DATAPATH=$ROOT/data
#RESULTPATH=$ROOT/results
DATASET=0

case $1 in 
	cifar) 	DATASET=cifar-10;;
	webspam) DATASET=webspam;;
	mnist) DATASET=mnist;;
	*) echo Invalid datapath, retry.
	exit 1
esac

OUTFILE=$(date +%Y%m%d-%H%M%S)_$1_$2_$3_$4_$5_$6

#Currently for SVD we're using the defaut method(0).
(time ./pmain $2 $ROOT/data/$DATASET/train_x 0 $3 $4) &> $OUTFILE-coreset_temp
CORESETTIME=${(grep "user	" $OUTFILE-coreset_temp)/user/}
#We take the user time into account.

./svm-train $ROOT/results/$DATASET/$OUTFILE $ROOT/results/$DATASET/$OUTFILE-svm_model $6


