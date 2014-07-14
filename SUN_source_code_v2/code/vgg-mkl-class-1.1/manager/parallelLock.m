function locked = parallelLock(id, varargin)
% PARALLELLOCK
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

jobopts.waitJobs    = [] ;
jobopts.numNodes    = 50 ;
jobopts.waitJobs    = [] ;
jobopts.freeMemMb   = 1900 ;

jobopts = vgg_argparse(jobopts, varargin) ;

% get who is calling me
drop = dbstack ;
callingMFile = drop(2).name ;
clear drop ;

useLockServer = 1 ;

% --------------------------------------------------------------------
%                                                      If not parallel
% --------------------------------------------------------------------

if ~ jobopts.parallelize
  locked = true ;
  return ;
end

% --------------------------------------------------------------------
%                                 If parallel, and taskId is specified
% --------------------------------------------------------------------

if ~isnan(jobopts.taskId)
  if ~useLockServer
    % This is a genuine parallel instance
    jobDir  = jobopts.jobDir ;
    jobId   = jobopts.jobId ;
    jobName = sprintf('j%d-%s', jobId, callingMFile) ;
    lockDir = fullfile(jobDir, jobName, 'locks') ;

    % try to acquire lock
    lockPath = fullfile(lockDir, sprintf('%06d.lock', id)) ;
    if exist(lockPath, 'file')
      fprintf('''%s'' already locked.\n', lockPath) ;
      locked = false ;
      return ;
    end
    system(sprintf('touch "%s"', lockPath)) ;
    fprintf('Acquired lock ''%s''.\n', lockPath) ;

  else
    locked = tinyLockClient('adam.robots.ox.ac.uk', '2000', sprintf('%d.%d',  jobopts.jobId, id)) ;
    return ;
  end

  locked = true ;
  return ;
end

% --------------------------------------------------------------------
%                              If parallel, and taskId is not specified
% --------------------------------------------------------------------

% This is the controller process only
locked = false ;
