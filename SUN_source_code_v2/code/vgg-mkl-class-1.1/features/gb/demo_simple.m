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




% Load two images
I1 = im2double(imread('data/1a.jpg'));
I2 = im2double(imread('data/2a.jpg'));


if (size(I1,3)>1),  % convert the image to grayscale
  I1 = mean(I1,3);
end


if (size(I2,3)>1),  % convert the image to grayscale
  I2 = mean(I2,3);
end

% compute channels using oriented edge energy
fbr1 = compute_channels_oe_nms(I1);
fbr2 = compute_channels_oe_nms(I2);

% try it using only one channel
% fbr1 = sum(fbr1,3);
% fbr2 = sum(fbr2,3);

% Parameters for the geometric blur descriptor
%
% the units are pixels, these numbers can be changed, 
% keeping in mind that the normalization step inside 
% get_descriptors might need to be changed as well.
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


% Actually extract Geometric Blur descriptors for each image
[descriptors1, pos1] = get_descriptors(fbr1,ndescriptors,rrep,alpha,beta,rs,nthetas);
[descriptors2, pos2] = get_descriptors(fbr2,ndescriptors,rrep,alpha,beta,rs,nthetas);

keyboard


% compute dissimilarity between features
diss = -comp_features(descriptors1,descriptors2);

sample_points = compute_sample_points(rs,nthetas);

% Interactively show the best match for points on the model,
% click on a point in the left (model) image to see its similarity
% to each point on the right image, with the best match marked with
% a magenta star
vis_gb_match(I1, pos1, I2, pos2, descriptors1, descriptors2, diss, sample_points)

