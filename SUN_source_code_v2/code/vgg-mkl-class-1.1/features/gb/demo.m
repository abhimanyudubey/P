% Demo code for matching roughly based on the procedure
% described in:
%
% "Shape Matching and Object Recognition using Low Distortion Correspondence"
% A. C. Berg, T. L. Berg, J. Malik
% CVPR 2005
%
% code Copyright 2005 Alex Berg
%
% questions -> Alex Berg aberg@cs.berkeley.edu



addpath descriptor_code
addpath correspondence_code
addpath feature_code
addpath visualization_code




% These data files contain oriented edge maps obtained using
% a version of Martin et al's edge detector, available at
% http://www.cs.berkeley.edu/projects/vision/grouping/segbench/
%
% There is a random choice for which points are used in the model
% run the code multiple times to see slightly different results.

% Helicopters example
dname1 = 'data/image_0037r.data.mat';
dname2 = 'data/image_0087r.data.mat';

% Ewers ( vases ) example
% dname1 = 'data/image_0012r.data.mat';
% dname2 = 'data/image_0063r.data.mat';


% Umbrella Flamingo example
% dname1 = 'data/image_0010r.data.mat';
% dname2 = 'data/image_0027r.data.mat';




% Load the data files
% These data files contain precomputed pb channels based on Martin
% et al's edge detector, a demo using a simple edge detector is
   % included in the other demo file, demo_simple

R1 = load(dname1);
I1 = R1.Iu;              % I1 is an image
fbr1 = im2double(R1.Au); % fbr1 contains the nonmax supressed
                         % oriented edge responses for I1
M = R1.M;                % M is the support mask of the object in I1
  
R2 = load(dname2);
I2 = R2.Iu;              % I2 is an image
fbr2 = im2double(R2.Au); % fbr2 contains the nonmax supressed
                         % oriented edg responses for I2

% In the example data files included there are eight orientations
% of edge responses, these are combined to make four channels
for i=1:4
  fbr1(:,:,i) = sum(fbr1(:,:,2*i-1:2*i),3);
  fbr2(:,:,i) = sum(fbr2(:,:,2*i-1:2*i),3);
end
fbr1 = fbr1(:,:,1:4);
fbr2 = fbr2(:,:,1:4);

% small signals are repressed to elminate some noise
% NOTE that the parameter here 0.2 is set for the pb boundary operator
% and that it probably should be changed for other edge operators...
tfbr1 = fbr1.*repmat(max(fbr1,[],3)>=0.2,[1 1 size(fbr1,3)]);
tfbr2 = fbr2.*repmat(max(fbr2,[],3)>=0.2,[1 1 size(fbr2,3)]);




% Parameters for the geometric blur descriptor
%
% the units are pixels, these numbers can be changed, but
% the normalization step inside get_descriptors might need to
% be adjusted
%

% rs are the radii for sample points
rs =      [0 4 8 16 32 50];

% nthetas are the number of samples at each radii
nthetas = [1 8 8 10 12 12];

% alpha is the rate of increase for blur
alpha = 0.5;

% beta is the base blur amount
beta = 1;


% Number of descriptors to extract per image
ndescriptors = 300;

% repulsion radius for rejection sampling
rrep = 5;


% Actually extract geometric blur descriptors for each image
% for the templates image we have a known support mask for the
% object, passed in as M, there is no mask for the second call to
% get_descriptors
[descriptors1, pos1] = get_descriptors(tfbr1,ndescriptors,rrep,alpha,beta,rs,nthetas,M);
[descriptors2, pos2] = get_descriptors(tfbr2,ndescriptors,rrep,alpha,beta,rs,nthetas);


% compute dissimilarity between features
diss = -comp_features(descriptors1,descriptors2);

sample_points = compute_sample_points(rs,nthetas);

% Interactively show the best match for points on the model,
% click on a point in the left (model) image to see its similarity
% to each point on the right image, with the best match marked with
% a magenta star
vis_gb_match(I1, pos1, I2, pos2, descriptors1, descriptors2, diss, sample_points)









% Actually do the quadratic assignment
nmp= 40;
match_alpha = 7;  noutliers = 10;  niters = 10;
id1 = zeros(size(descriptors1,1),1);
id2 = zeros(size(descriptors2,1),1);
selected1 = ones(size(descriptors1,1),1)>0;
[RRR,H,c,x,ind_i,ind_j] = compute_correspondence_wrapper(descriptors1,...
                                     descriptors2,pos1,pos2,selected1,...
                                     match_alpha,niters,nmp,noutliers,id1, id2,...
                                     2, 8, 0.1, 5);
ox = x;
fx = find(x);
fx = fx(ind_j(fx)>0);
p1 = pos1(ind_i(fx),:);
p2 = pos2(ind_j(fx),:);

clear p22
for i=1:length(fx),
  [mv,mi]=min(diss(ind_i(fx(i)),:));
  p22(i,:) = pos2(mi(1),:);
end


[p1m,coeffs] = warp_points(p1,p2,2000,pos1);
p1m = p1m';




% display the correspondence
% visualization parameters
ms = 20;
ms_big = 40;

clf
subplot(2,2,1);
imagesc(I1);
hold on
plot_points(round(0.1*sum(p1,2)),p1(:,2),p1(:,1),ms_big,ms);
%plot_points(round(0.1*sum(pos1,2)),pos1(:,2),pos1(:,1),ms_big,ms);
title('Template with model points')


subplot(2,2,2);
imagesc(I2);
hold on
plot_points(round(0.1*sum(p1,2)),p2(:,2),p2(:,1),ms_big,ms);
title('Target with matching points')


subplot(2,2,3);
imagesc(I1);
hold on
plot_points(round(0.1*sum(pos1,2)),pos1(:,2),pos1(:,1),ms_big,ms);

title('Template with all points')


subplot(2,2,4);
imagesc(I2);
hold on
plot_points(round(0.1*sum(pos1,2)),p1m(:,2),p1m(:,1),ms_big,ms);
title('Target with interpolated match')

axis image
subplot(2,2,3)
axis image
subplot(2,2,2)
axis image
subplot(2,2,1)
axis image
