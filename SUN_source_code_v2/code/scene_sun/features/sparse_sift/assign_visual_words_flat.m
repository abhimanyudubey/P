function [histogram] = assign_visual_words_flat(sifts, cluster_centers)
%new version treats it more like a flat hierarchy
%
%same function should work for mser and hesaff sifts
%for the first level, I can just do one distsqr_fast
%   cluster_centers1              128x1000             204800  double
%   

%since this vocabulary only has 1000 entries, we will not store it as a
%sparse array.

%constants for the soft assignment
r     = 10; %nearest neighbors to weight between
sigma = 65; %distance to use in weighting

top_dists = distSqr_fast(sifts', cluster_centers);
[top_dists, top_inds] = sort(top_dists, 2); %sorts each row independently

% all_lvl1_dists = zeros(1,r);
% total_weights  = zeros(1,r);

histogram = zeros(size(cluster_centers,2),1); %one entry for each cluster center

for i = 1:size(sifts,1)
    cur_dists = top_dists(i,1:r);
    cur_inds  = top_inds( i,1:r);

%     all_lvl1_dists = all_lvl1_dists + cur_dists;
    
    weights = exp( -cur_dists ./ (2*sigma.^2));
    weights = weights ./ sum(weights);
    
%     total_weights = total_weights + weights;
    
    histogram(cur_inds) = histogram(cur_inds)' + weights;

end

histogram = single(histogram);

% histogram(histogram < 0.05) = 0;

% [inds,tmp,values] = find(histogram);
% inds = uint16(inds);
% values = uint16(values * 100);

% all_lvl1_dists ./ size(sifts,1)
% total_weights ./ size(sifts,1)
% sum(histogram)
% sum(histogram > 0.05)
% pause

