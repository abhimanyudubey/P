function e = vgg_kmeans_cluster_err(p, X)

% VGG_KMEANS_CLUSTER_ERR A function
%               ...

% Author: Andrew Fitzgibbon <awf@robots.ox.ac.uk>
% Date: 30 Nov 02

dimension = size(X,1);
npoints = size(X, 2);
nclusters = length(p) / dimension;

cluster_centres = reshape(p, dimension, nclusters);

if 0
  hold off
  scatter(X', '.');
  hold on
  scatter(cluster_centres', 'ro');
  axis([0 1 0 1])
  axis square
  drawnow
end

% Compute distance from each point to each cluster center
distances = zeros(nclusters, npoints);
for m = 1:nclusters
  cc = cluster_centres(:,m);
  diff = repmat(cc, 1, npoints) - X;
  if 1
    distances(m,:) = sum(diff.^2,1);  % squared
  else
    distances(m,:) = sqrt(sum(diff.^2,1));  % absolute
  end
end

e = sum(min(distances));
