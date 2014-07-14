#!/bin/bash
#
# Call:
#   qsub_matlab.sh MFILE
#

[ -r ~/.bashrc ] && . ~/.bashrc
[ -r ~/.bash_profile ] && . ~/.bash_profile

export TERM=vt100

echo "*************** JOB BEGINS *****************"
echo "Task id: $SGE_TASK_ID"
echo "Job id: $JOB_ID"
echo "Work dir: $SGE_O_WORKDIR"
echo "Matlab command: $1"
echo "Host: " `hostname`
echo "Date started:" `date`

cd "$SGE_O_WORKDIR" ;

matlab -nosplash -nojvm -nodisplay <<EOF
setup ;
jobId=$SGE_TASK_ID ;
sgeJobId=$JOB_ID ;
sgeTaskId=$SGE_TASK_ID ;
$1 ;
exit ;
EOF
echo "Date finished:" `date`
echo "**************** JOB ENDS ******************"
echo
