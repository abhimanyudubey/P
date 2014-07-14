function [K K_test]= kernel(X, Y, kernelName, wantK)

%   KL2    sum X .* Y
%   KL1    sum min (X, Y)
%   KCHI2  2 * sum (X .* Y) ./ (X + Y)
%   KHELL  (X .* Y) .^ 0.5

if (wantK || strcmp(kernelName,'echi2')==1 || strcmp(kernelName,'emd')==1 || strcmp(kernelName,'rbf')==1)
switch kernelName
    case 'emd'
        K = emd_dist(X,X);
        mu= 1 ./ mean(K(:)) ;
        K = exp(- mu * K) ;    
    case 'echi2'
        K = alldist2(X, 'chi2') ;
        mu= 1 ./ mean(K(:)) ;
        K = exp(- mu * K) ;
    case 'kl1' %hist_intersect
        K = vl_alldist2(X, 'kl1');
    case'kl2'
        K = X' * X ;
    case 'kchi2'
        K = vl_alldist2(X, 'kchi2') ;
    case 'gb'
        K = gbDistance(X, X);
    case 'rbf'
        X = X';
        norm1 = sum(X.^2,2);
        norm2 = sum(X.^2,2);
        dist = (repmat(norm1 ,1,size(X,1)) + repmat(norm2',size(X,1),1) - 2*X*X');
        mu=sqrt(mean(dist(:))/2);
        K = exp(-0.5/mu^2 * dist);
end
if ~isempty(find(isnan(K)))
    disp('something element is NaN in the K matrix');
    K(find(isnan(K)))=10^20;
end
else
    K = [];
end

switch kernelName
    case 'emd'
        K_test = emd_dist(X, Y);
        K_test = exp(- mu * K_test) ;
    case 'echi2'
        K_test = vl_alldist(X, Y, 'chi2') ;
        K_test = exp(- mu * K_test) ;
    case 'kl1' %hist_intersect
        K_test = vl_alldist2(X, Y, 'kl1');
    case'kl2'
        K_test = X' * Y ;
    case 'kchi2'
        K_test = vl_alldist2(X, Y, 'kchi2') ;
    case 'gb'
        K_test = gbDistance(X, Y);
    case 'rbf'
        Y = Y';
        norm1 = sum(X.^2,2);
        norm2 = sum(Y.^2,2);
        dist = (repmat(norm1 ,1,size(Y,1)) + repmat(norm2',size(X,1),1) - 2*X*Y');
        %mu=sqrt(mean(dist(:))/2);
        K_test = exp(-0.5/mu^2 * dist);
end

if ~isempty(find(isnan(K_test)))
    disp('something element is NaN in the K_test matrix');
    K_test(find(isnan(K_test)))=10^20;
end