function final_texton_histogram = assign_weighted_map(texton_map, weight_map, nbClusters)

if(~exist('nbClusters', 'var'))
    nbClusters = 512;
end

if(~isfloat(texton_map))
    fprintf('texton map needs to be float\n')
    return
end

if(~isfloat(weight_map))
    fprintf('weight map needs to be float\n')
    return
end


texton_vect = texton_map(:);
weight_vect = weight_map(:);

points = [texton_vect weight_vect];

texton_edges = 1:nbClusters;
weight_edges = 0:(1/256):1;

%2d joint histogram
texton_histogram = histnd(points, texton_edges, weight_edges);

% sum(texton_histogram,1)
% sum(texton_histogram,2)
% for i = 2: size(texton_histogram,2)
%     sum(texton_histogram(:,i))
%     sum( weight_vect > weight_edges(i-1) & weight_vect <= weight_edges(i))
% end

% sum(sum(sum(sum(color_hist))))
% texton_histogram = texton_histogram(1:end-1, 1:end); %the last bin in each dimension is empty, except the weight dimension
% sum(sum(sum(sum(color_hist))))

%now we want to marginalize over the last dimension
final_texton_histogram = zeros(nbClusters, 1);

%skipping first index because that's mostly zero confidence
for i = 2:size(texton_histogram,2)-1
    final_texton_histogram  = final_texton_histogram + ((i-0.5)/size(texton_histogram,2)) * texton_histogram(:,i);
end
final_texton_histogram = final_texton_histogram + texton_histogram(:,end); %the full confidence ones
 
final_texton_histogram = single(final_texton_histogram);

% sum(final_texton_histogram)
% sum(sum(weight_map))
% pause
