function [texton_map] = calc_texton_map(input_image, filterBank, clusterCenters)
%jhhays 4/21/2009
%computes texton map (nearest neighbor texton for each pixel).
%preserves the input resolution.

%image should be [0 1], color, double

% assumption -- the calling function already added this
% if(exist('fbCreate')~=2)
%     %if the path for fbCreate wasn't already added, try to do so now
%     addpath('berk_textons_lib/')
% end

%fprintf('\n Calculating texton map\n')
display = 0;

if(size(input_image,3) > 1)
    input_image = rgb2gray(input_image);
end
[h,w] = size(input_image);

%fprintf('  Filtering image...'); t1 = clock;
filteredImg = fbRun(filterBank, input_image);
%fprintf('done in %fs.\n', etime(clock,t1));

if(display)
    figure(738)
    imagesc(filteredImg{6,2})
end

% stack it into a nbPixel * nbDims vector
filteredImg = cellfun(@(x) reshape(x, [size(x,1)*size(x,2) 1]), filteredImg, 'UniformOutput', 0);
filteredImg = reshape(filteredImg, [1, size(filteredImg,1)*size(filteredImg,2)]);
filteredImg = [filteredImg{:}];

% nbClusters = size(clusterCenters,2); %the cluster centers were passed in as a parameter

% find nearest neighbors for each pixel, and assign its index
%fprintf('  Assigning textons...'); t0 = clock;

% [ind, d] = vgg_nearest_neighbour(filteredImg', clusterCenters);
ind = MEXfindnearestl2(filteredImg', clusterCenters);

% dis = zeros([1 size(clusterCenters,2)]);
% for i=1:size(filteredImg,1)
%     cur_f_vec = filteredImg(i,:)';
%     for j=1:size(clusterCenters,2)
%         dis(j) = norm(cur_f_vec-clusterCenters(:,j));
%     end
%     [ind(i) d(i)]=min(dis);
% end


texton_map = reshape(ind, [h w]);
%fprintf('done in %fs.\n', etime(clock,t0));

if(display)
    figure(739)
    imagesc(texton_map)
    truesize
    pause
end

% texton_histogram = histc(texton_map(:), 1:nbClusters);
% texton_histogram = single(texton_histogram);

