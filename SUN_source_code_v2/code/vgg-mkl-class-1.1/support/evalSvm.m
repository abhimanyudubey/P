function scores = evalSvm(svm, trainHists, testHists)
% EVALSVM  Evaluate SVM model
%   SCORES = EVALSVM(SVM, TRAINHISTS, TESTHISTS)
%
%   TRAINHISTS nad TESTHISTS are expected int SOA format.
%
%   For compressed kernels, pass TRAINHISTS = [].
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

K = length(svm.kernels) ;
scores = [] ;

for ki=1:K
  name = svm.kernels(ki).histName ;
  ts = testHists.(name) ;

  if isfield(svm.kernels(ki), 'compr') & ~isempty(trainHists.(name))
    warning(['compressed kernel ''%s'' not being used becase ' ...
             'TRAINHISTS is not empty'], svm.kernels(ki).name) ;
  end

  if isempty(trainHists.(name))

    % Zeroed kernel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if isempty(testHists.(name))
      continue ;
    end

    % Compressed kernel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    aff = pwlKerEval(svm.kernels(ki).compr.A,  ...
                     svm.kernels(ki).compr.B,  ...
                     svm.kernels(ki).compr.th, ...
                     ts) ;
  else
    % Plain kernel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tr = trainHists.(name)(:,svm.svind) ;
    aff = calcKernel(svm.kernels(ki), tr, ts, svm.alphay) ;
  end

  if isempty(scores)
    scores = svm.d(ki) * aff ;
  else
    scores = scores + svm.d(ki) * aff ;
  end
end

scores = scores + svm.b ;
