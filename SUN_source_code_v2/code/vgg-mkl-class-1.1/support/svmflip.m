function svm = svmflip(svm, labels)
% SVMFLIP  Flip SVM
%   Uses a simple heuristic to flip a binary SVM so that positive
%   labels correspond to positive socres.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if ~isempty(setdiff(unique(labels), [-1 +1]))
  error('LABELS must be binrary') ;
end

labels = labels(svm.svind) ;

ap = mean(svm.alphay(labels > 0)) ;
am = mean(svm.alphay(labels < 0)) ;

if ap < am
  fprintf('svmflip: SVM appears to be flipped. Adjusting.\n') ;
  svm.alphay = - svm.alphay ;
  svm.b      = - svm.b ;
end
