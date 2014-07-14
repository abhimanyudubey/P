function vocab = vggCluster(featOpts, descrs)
% IKMEANSCLUSTER  K-means quantization
%
%  Author:: Andrea Vedaldi

if ~isa(descrs, 'double') ; descrs = double(descrs) ; end

vocab = vgg_kmeans(descrs, featOpts.vocabSize, 'verbose', 1) ;
