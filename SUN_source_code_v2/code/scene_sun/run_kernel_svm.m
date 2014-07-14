clear;
addpath(genpath('../'));
vl_setup('noprefix');
n=0;

%% for 8 classes dataset
% conf.featPath   = '../../data/scene_8class/feature/';
% conf.kernelPath = '../../data/scene_8class/kernel/';
% conf.resultPath = '../../data/scene_8class/result/';
% conf.splitFile  = '../../data/scene_8class/split10.mat';
% conf.max_num_train = 100;
% conf.max_num_test = inf;
% conf.imageDir = true;
%% for 15 classes dataset
  conf.featPath   = '../../data/scene_15class/feature/';
  conf.kernelPath = '../../data/scene_15class/kernel/';
  conf.resultPath = '../../data/scene_15class/result/';
  conf.splitFile  = '../../data/scene_15class/split10.mat';
  conf.max_num_train = 100;
  conf.max_num_test = inf;
  conf.imageDir = true;
%% for SUN 397
% conf.featPath   = '../../data/scene_397class/feature/';
% conf.kernelPath = '../../data/scene_397class/kernel/';
% conf.resultPath = '../../data/scene_397class/result/';
% conf.splitFile  = '../../data/scene_397class/split10.mat';
% conf.max_num_train = 50;
% conf.max_num_test = 50;
% conf.imageDir = true;

%% load split file
load(conf.splitFile);

%% kernel configuration

% 'gist'
n=n+1;
parameterCell{n}.featureName  = 'gist';
parameterCell{n}.kernelName   = 'echi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'gist';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'gist';
parameterCell{n}.kernelName   = 'kl2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'gist';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'gist';
parameterCell{n}.kernelName   = 'rbf';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;

% 'hog2x2'
n=n+1;
parameterCell{n}.featureName  = 'hog2x2';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'hog2x2';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'hog2x2';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'hog2x2';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;

% 'tiny_image'
n=n+1;
parameterCell{n}.featureName  = 'tiny_image';
parameterCell{n}.kernelName   = 'rbf';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'tiny_image';
parameterCell{n}.kernelName   = 'echi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;

% 'lbp'
n=n+1;
parameterCell{n}.featureName  = 'lbp';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'lbp';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'lbp';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'lbp';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'lbp';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;


% 'lbphf'
n=n+1;
parameterCell{n}.featureName  = 'lbphf';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'lbphf';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'lbphf';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'lbphf';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;



% 'denseSIFT'
n=n+1;
parameterCell{n}.featureName  = 'denseSIFT';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'denseSIFT';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'denseSIFT';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'denseSIFT';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;



% line_hists
n=n+1;
parameterCell{n}.featureName  = 'line_hists';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'line_hists';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;


% 'gistPadding'
n=n+1;
parameterCell{n}.featureName  = 'gistPadding';
parameterCell{n}.kernelName   = 'rbf';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;



%%

% sparse_sift
n=n+1;
parameterCell{n}.featureName  = 'sparse_sift';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'sparse_sift';
parameterCell{n}.kernelName   = 'kl1';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;

% 'ssim'
n=n+1;
parameterCell{n}.featureName  = 'ssim';
parameterCell{n}.kernelName   = 'echi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'ssim';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'ssim';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'ssim';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'ssim';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;


% 'texton'
n=n+1;
parameterCell{n}.featureName  = 'texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;

%%


% 'geo_map8x8'
n=n+1;
parameterCell{n}.featureName  = 'geo_map8x8';
parameterCell{n}.kernelName   = 'rbf';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;

% 'geo_texton'
n=n+1;
parameterCell{n}.featureName  = 'geo_texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'geo_texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'geo_texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'geo_texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;

% texton histogram

n=n+1;
parameterCell{n}.featureName  = 'geo_texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = true;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'geo_texton';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = true;
parameterCell{n}.normalize    = true;

% 'geo_color'
n=n+1;
parameterCell{n}.featureName  = 'geo_color';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = true;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'geo_color';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = true;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'geo_color';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'geo_color';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = false;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;
n=n+1;
parameterCell{n}.featureName  = 'geo_color';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = false;
n=n+1;
parameterCell{n}.featureName  = 'geo_color';
parameterCell{n}.kernelName   = 'kchi2';
parameterCell{n}.weighted     = true;
parameterCell{n}.bow          = false;
parameterCell{n}.normalize    = true;


%% individual kernels
nLeng = length(parameterCell);
disp(num2str(nLeng));
for i=1:length(split)
    for j=1:nLeng
        disp(['split ' num2str(i) ' kernel ' num2str(j)]);
        para = parameterCell{j};
        para.splitID = i;
        perf(j,:,i) = compute_kernel_svm(para,conf);
        save([conf.resultPath 'perf.mat'],'perf');
    end
end
avg_perf = mean(perf,3);
avg_perf = avg_perf(:,end:-1:1);

%% plot results
[foo, ndx] = sort(avg_perf(:,end), 'descend');
clear plotLabel;
for i=1:nLeng
    outName = get_kernel_filename(parameterCell{i});
    outName = regexprep(outName,'_',' ');
    plotLabel{i} = sprintf('[%2.1f] %s',avg_perf(i,end),outName) ;
end
plot((avg_perf(ndx,:))');
legend(plotLabel(ndx));
grid on
axis('square')
% set(gca, 'XScale', 'log')
% set(gca, 'XTick', (kRange))
% set(gca, 'XTickLabel', kRange)
% axis([0 max(kRange)+1 0 25 ])

%% weighting
featMap =containers.Map();
for i=1:nLeng
    featMap(parameterCell{i}.featureName)=i;
    parameterCell{i}.weight = 0;
end
for i=1:nLeng
   if (avg_perf(i,end)>avg_perf(featMap(parameterCell{i}.featureName),end))
       featMap(parameterCell{i}.featureName) = i;
   end
end
median_avg_perf = median(avg_perf(:,end));
disp(['median perf ' num2str(median_avg_perf)]);
for i=cell2mat(values(featMap))
    parameterCell{i}.weight = (avg_perf(i,end)/median_avg_perf)^4;
    disp(['combining ' num2str(avg_perf(i,end)) ' ' num2str(parameterCell{i}.weight) ' ' get_kernel_filename(parameterCell{i})])
end
    
%% combine kernel configuration
n=0;
n=n+1;
paraCombine{n}.featureName  = 'all';
paraCombine{n}.kernelName   = 'combine';
paraCombine{n}.weighted     = false;
paraCombine{n}.bow          = false;
paraCombine{n}.normalize    = true;
n=n+1;
paraCombine{n}.featureName  = 'all';
paraCombine{n}.kernelName   = 'combine';
paraCombine{n}.weighted     = false;
paraCombine{n}.bow          = false;
paraCombine{n}.normalize    = false;

%% combine kernel and svm
disp('combining kernels ...');
nLengCombine = length(paraCombine);
disp(num2str(nLeng));
for i=1:length(split)
    for j=1:nLengCombine
        disp(['split ' num2str(i) ' kernel ' num2str(j)]);
        para = paraCombine{j};
        para.splitID = i;
        combine_kernel(parameterCell,conf,para);
        perf_comb(j,:,i) = compute_kernel_svm(para,conf);
    end
end
avg_perf_comb = mean(perf_comb,3);

avg_perf_comb = avg_perf_comb(:,end:-1:1);
fprintf('combine kernel accuracy (normalized)   = %2.1f\n',avg_perf_comb(1,end));
fprintf('combine kernel accuracy (unnormalized) = %2.1f\n',avg_perf_comb(2,end));

