% CAL_DEMO  Run a Caltech-101 experiment
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

magicMasterExpNum = 1 ;
do_classMulti     = 1 ;

expNumRange = [1 6 2 3 4 5] ;
numTrainRange   = [15 30] ;
randomSeedRange = [1 2 3] ;

% --------------------------------------------------------------------
%                                                 Combine all classess
% --------------------------------------------------------------------

magicSmallData = false ;
for magicMasterExpNum = expNumRange
  for magicMasterNumTrain = numTrainRange
    for magicMasterRandomSeed = randomSeedRange
      cal_conf ;
      if do_classMulti, cal_classMulti ; end
    end
  end
end

% --------------------------------------------------------------------
%                                                       Generate plots
% --------------------------------------------------------------------

roidb = getRoiDb(conf.gtRoiDbPath) ;
summary = [] ;
expLabels = {} ;

for exi = 1:length(expNumRange)
  magicMasterExpNum = expNumRange(exi) ;

  for nti = 1:length(numTrainRange)
    magicMasterNumTrain = numTrainRange(nti) ;

    testConf     = {} ;
    diagTestConf = [] ;
    mi = 0 ;

    for magicMasterRandomSeed = randomSeedRange
      cal_conf ;
      commDir = fullfile(conf.trainDir, 'COMMON', conf.expPrefix) ;
      confPath = fullfile(commDir, 'conf') ;

      % load confusion matrix
      fprintf('Loading confusion matrix %s\n', confPath) ;
      tmp = load(confPath) ;
      testConf{end+1} = tmp.testConf ;

      % normalize confusion matrix by rows
      numClasses = size(testConf{end}, 1) ;
      testConf{end} = testConf{end} ./ ...
          (sum(testConf{end},2) * ones(1, numClasses)) ;

      % extract diagonal
      diagTestConf = [diagTestConf, diag(testConf{end})] ;
    end

    summary(nti,exi).label = sprintf('%s %d', conf.expPrefix, magicMasterNumTrain) ;
    summary(nti,exi).meanAccuracy = 100 * mean(mean(diagTestConf,1)) ;
    summary(nti,exi).stdAccuracy  = 100 * std(mean(diagTestConf,1)) ;

    mean(diagTestConf)

    figure(2) ; clf ;
    muDiagTestConf = mean(diagTestConf, 2) ;
    stdDiagTestConf = std(diagTestConf, 1, 2) ;
    [drop,perm] = sort(muDiagTestConf, 'descend') ;
    errorbar(100*muDiagTestConf(perm), 100*stdDiagTestConf(perm)) ;

    title(sprintf('%s: accuracy for %d training samples: %.2f%% \\pm %.2f%%', ...
                  conf.expPrefix, ...
                  magicMasterNumTrain, ...
                  summary(nti,exi).meanAccuracy, ...
                  summary(nti,exi).stdAccuracy)) ;
    xlim([1 102]) ; xlabel('Class ID') ;
    ylim([1 100]) ; ylabel('Accuracy (%)') ;
    grid on ;
    printsize(.7) ;

    outPrefix = fullfile('data', ...
                         sprintf('cal-%d-%s' ,...
                                 magicMasterNumTrain, ...
                                 conf.expPrefix)) ;
    print(gcf,'-depsc2', [outPrefix '-acc.eps']) ;
    %system(sprintf('convert -density 300 %s %s', [outPrefix '-acc.eps'], [outPrefix '-acc.png'])) ;

    txts = fieldnames(roidb.classes) ;
    fid = fopen([outPrefix '-info.txt'], 'w') ;
    for ci=1:numClasses
      fprintf(fid, '%22s: %5.1f %% pm %5.1f\n', ...
              txts{perm(ci)}, ...
              100 * muDiagTestConf(perm(ci)), ...
              100 * stdDiagTestConf(perm(ci))) ;
    end
    fclose(fid) ;
  end

  expLabels{exi} = conf.expPrefix(strfind(conf.expPrefix, '-')+1:end) ;
  if isempty(expLabels{exi}), expLabels{exi} = 'MKL'; end
end

figure(3) ; clf ;
h = [] ;
leg = {} ;
styles = {'ro', 'go'} ;
for nti = 1:length(numTrainRange)
  hold on ;
  h(end+1) = errorbar(1:length(expNumRange), ...
                      [summary(nti,:).meanAccuracy], ...
                      [summary(nti,:).stdAccuracy], styles{nti}) ;
  leg{end+1} = sprintf('%d training', numTrainRange(nti)) ;
end
legend(h, leg{:}, 'location', 'northeast') ;
title('Feature comparison') ;
set(gca,'xtick', 1:length(expLabels), 'xticklabel', expLabels) ;
set(gca,'YGrid', 'on') ;
ylabel('Accuracy (%)') ;

outPrefix = fullfile('data', 'cal-summary') ;
printsize(.6) ;
%print(gcf,'-dpng', '-r300', [outPrefix '.png']) ;
print(gcf,'-depsc2', [outPrefix '.eps']) ;
system(sprintf('convert -density 300 %s %s', [outPrefix '.eps'], [outPrefix '.png'])) ;
