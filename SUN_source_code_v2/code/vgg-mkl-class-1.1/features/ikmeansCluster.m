function vocab = ikmeansCluster(featOpts, descrs)
% IKMEANSCLUSTER  K-means quantization
%
%  Author:: Andrea Vedaldi

vocab = vl_ikmeans(descrs, featOpts.vocabSize, 'method', 'elkan', 'verbose') ;
