function kerdb = calcKernelDb(typeNames, featNames, featOpts) ;
% CALCKERNELDB  Get the kernels database
%  KERDB = CALCKERNELDB(TYPENAMES, FEATNAMES, FEATOPTS) returns a
%  struct array describing the available SVM kernels.
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

kerdb = []  ;
nk = 0 ;
for ti=1:length(typeNames)
  typeName = typeNames{ti} ;

  for fi=1:length(featNames)
    featName = featNames{fi} ;
    pyrLevels = featOpts.(featName).pyrLevels ;

    if isempty(pyrLevels)
      nk = nk + 1 ;
      kerdb(nk) .type     = typeName ;
      kerdb(nk) .feat     = featName ;
      kerdb(nk) .pyrLevel = [] ;
      kerdb(nk) .histName = featName ;
      kerdb(nk) .name     = sprintf('%s_%s', ...
                                    typeName, ...
                                    featName) ;
      continue ;
    end

    for pyrLevel=pyrLevels
      nk = nk + 1 ;
      kerdb(nk) .type      = typeName ;
      kerdb(nk) .feat      = featName ;
      kerdb(nk) .pyrLevel  = pyrLevel ;
      kerdb(nk) .histName  = sprintf('%s_L%d', ...
                                     featName, ...
                                     pyrLevel  ) ;
      kerdb(nk) .name      = sprintf('%s_%s_L%d', ...
                                     typeName,    ...
                                     featName,    ...
                                     pyrLevel     ) ;

    end
  end
end
