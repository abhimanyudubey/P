function [jobId, taskId] = parallelDriver(varargin)
% PARALLELDRIVER
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

jobopts.parallelize = 0 ;
jobopts.jobId       = NaN ;
jobopts.taskId      = NaN ;
jobopts.jobDir      = '' ;
jobopts.needsJava   = false ;

jobopts.waitJobs    = [] ;
jobopts.numNodes    = 10 ;
jobopts.freeMemMb   = 1900 ;

jobopts = vl_argparse(jobopts, varargin{:}) ;

% waiting list
job_qsub_opts = [] ;

if ~isempty(jobopts.waitJobs)
  jobList = sprintf('%d,', jobopts.waitJobs) ;
  job_qsub_opts = [job_qsub_opts ...
                   ' -hold_jid ' jobList(1:end-1)] ;
end

if jobopts.freeMemMb > 0
  job_qsub_opts = [job_qsub_opts ...
                   sprintf(' -l mem_free=%gm', jobopts.freeMemMb) ...
                   sprintf(' -l vf=%gm', jobopts.freeMemMb)] ;
end

% get who is calling me
drop = dbstack ;
callingMFile = drop(2).name ;
clear drop ;

taskId = jobopts.taskId ;
jobId  = jobopts.jobId ;

% --------------------------------------------------------------------
%                                                      If not parallel
% --------------------------------------------------------------------

if ~ jobopts.parallelize
  fprintf('>>>>> %s <<<<<\n', callingMFile) ;
  return ;
end

% --------------------------------------------------------------------
%                                 If parallel, and taskId is specified
% --------------------------------------------------------------------

if ~isnan(jobopts.taskId)
  return ;
end

% --------------------------------------------------------------------
%                             If parallel, and taskId is not specified
% --------------------------------------------------------------------

tic ;

outputDir  = fullfile(jobopts.jobDir, sprintf('j$JOB_ID-%s', callingMFile)) ;
stdErrPath = fullfile(outputDir, 'err$TASK_ID.txt') ;
stdOutPath = fullfile(outputDir, 'out$TASK_ID.txt') ;

%stdErrPath = fullfile(jobopts.jobDir, 'err$JOB_ID-$TASK_ID.txt') ;
%stdOutPath = fullfile(jobopts.jobDir, 'out$JOB_ID-$TASK_ID.txt') ;

matCmd = sprintf(['jobName=sprintf(''j%%d-%s'',sgeJobId);' ...
                  'jobPath=fullfile(''%s'', jobName);' ...
                  'load(fullfile(jobPath,''conf''));' ...
                  'disp(conf);' ...
                  '%s(conf,''parallelize'',1,''jobId'',sgeJobId,' ...
                  '  ''taskId'',sgeTaskId,''jobDir'',''%s'');'], ...
                 callingMFile, ...
                 jobopts.jobDir, ...
                 callingMFile, ...
                 jobopts.jobDir) ;

if jobopts.needsJava
  scriptName = 'qsub_matlab_java.sh' ;
else
  scriptName = 'qsub_matlab.sh' ;
end

cmd = sprintf(['qsub '                                  ...
               '-h '                                    ...
               '-t 1-%d '                               ...
               '-o ''%s'' '                             ...
               '-e ''%s'' '                             ...
               '-N ''%s'' '                             ...
               ' %s '                                   ...
               '%s "%s"'],                              ...
              jobopts.numNodes,                         ...
              stdOutPath, stdErrPath,                   ...
              callingMFile,                             ...
              job_qsub_opts,                            ...
              scriptName,                               ...
              matCmd) ;

[s,w] = system(cmd) ;
if s ~= 0
  fprintf('Offending command:\n%s\n\n', cmd) ;
  error(w) ;
end

tmp = regexp(w, ' ([0-9]*)\.', 'tokens') ;
jobId = sscanf(tmp{1}{1}, '%d') ;
taskId = NaN ;

% now can setup actual output directory
jobName = sprintf('j%d-%s', jobId, callingMFile) ;
outputDir = fullfile(jobopts.jobDir, jobName) ;
ensuredir(outputDir) ;
%ensuredir(fullfile(outputDir, 'locks')) ;
confPath   = fullfile(outputDir, 'conf.mat') ;
evalin('caller', sprintf('ssave(''%s'', ''conf'')', confPath)) ;

fid = fopen(fullfile(outputDir, 'SGE_SUB_TIME'),'w') ;
fwrite(fid, datestr(now)) ;
fclose(fid) ;

fid = fopen(fullfile(outputDir, 'SGE_CMD'),'w') ;
fwrite(fid, cmd) ;
fclose(fid) ;

fid = fopen(fullfile(outputDir, 'SGE_ID'),'w') ;
fwrite(fid, tmp{1}{1}) ;
fclose(fid) ;

fprintf('Spawning job array ''%s'' with %d instances.\n',  ...
        outputDir, jobopts.numNodes) ;
fprintf('\tSGE job id: %d\n', jobId) ;
fprintf('\tqsub additional options: ''%s''.\n', job_qsub_opts) ;

% remove hold
cmd = sprintf('qalter -h U %d', jobId) ;
[s,w] = system(cmd) ;
if s ~= 0
  fprintf('Offending command:\n%s\n\n', cmd) ;
  error(w) ;
end

% this will ensure that each job directory has a different time stamp
%pause(max(1.5 - toc, 0)) ;
