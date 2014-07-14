function [angle_hist, length_hist] = calc_line_feats_img(lines, confidences, mask, segmentation, npixels)
%each row of lines is [x1 x2 y1 y2 theta r];

%The line statistics are no longer normalized based on image size, although
%images were downsampled to at most 800 pixels in their max dimension
%before the line segments were computed.  As a form of normalization, the
%line feature should be scaled by the image area (max dimension 800).

%for 800 by XXX images, or images that were originally larger, this will do nothing. 
%but for smaller images this will magnify the coordinates
% original_image_size = original_image_size(1:2);
% original_image_size = original_image_size / max(original_image_size);
% original_image_size = round(original_image_size * 800);

%do we need a double normalization?  As images get smaller, the detected
%lines don't just get smaller they get rarer.  But we don't entirely want
%to normalize for image size, do we?  We don't want tiny images from the
%database matching big queries.  As an image shrinks, do we want it to
%maintain the same features?

%yes, we do want double normalizatin.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% line segment stuff

% lines(:, 1:2) = lines(:, 1:2) * original_image_size(1);
% lines(:, 3:4) = lines(:, 3:4) * original_image_size(1);
% 
% lines(:, 1:2) = lines(:, 1:2) + original_image_size(2)/2;
% lines(:, 3:4) = lines(:, 3:4) + original_image_size(1)/2;

%histogram of angle occurance (weighted by length and confidence)
if(size(mask,3) > 1)
    fprintf('Error, mask is not grayscale\n')
    return
end

original_image_size = size(mask);

original_area = original_image_size(1) * original_image_size(2);
% original_area = 800*600

%this normalization accounts for the fact that the same lines in a
%downsampled image are necessarily shorter.  This exactly cancels that
%effect.
length_normalization = 800/max(original_image_size);


%this normalization accounts for the fact that in a downsampled image,
%fewer lines are detected (especially shorter lines).  This normalization
%is more wishy washy.
area_normalization   = sqrt(sqrt(original_area));
% area_normalization = 1

%if the image is incomplete, we want to weight the existing areas more.
%(assumption that the image statistics are uniform)
masked_area     = sum(sum(mask));
% original_area = original_area - masked_area;
valid_percentage = (original_area - masked_area) / original_area;

%the images are all reduced to max dimension 800 before the line
%computation.

%don't allow confidence more than 800;
confidences(confidences > 800) = 800;

if(size(lines,1) == 0)
    fprintf('This image has no lines\n');
    angle_hist  = zeros(1,200);
    length_hist = zeros(1,30);
    return
end

%non-normalized lengths
lengths = sqrt((lines(:,1) - lines(:,2)).^2 + (lines(:,3) - lines(:,4)).^2);

%this would be the only normalization necessary if scale didn't affect the
%detection rate of lines, but it does, so we'll need to normalize even more
%strongly than this.  This normalization scales the lengths as if the image
%had been max dimension 800.
lengths = lengths * length_normalization;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% angle stuff

angles  = lines(:,5);
%make sure all angles are [0, 2pi]
for i = 1:size(angles,1)
    if( angles(i) < 0 )
        angles(i) = angles(i) + 2*pi;
    end
end
angles = angles ./ pi;   %there is an implied pi in the units now.

%bin angle indices
t = .005:.005:2;  %400 bins
bins = zeros(size(t));

for i = 1:size(angles,1)
    current_angle = angles(i);
    current_angle = floor(current_angle * 200);
    if(current_angle > 399)
        current_angle
        current_angle = current_angle - 1;
    end
    %+ 1 offset because zero angle goes in bin 1
%     bins(current_angle + 1) = bins(current_angle + 1) + lengths(i);
    bins(current_angle + 1) = bins(current_angle + 1) + lengths(i) * sqrt(confidences(i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%

%problem- since the angles only lie in two quadrants, there is a big
%discontinuity for near-horizontal edges.  A small angle change leaves them
%in completely opposite sides of the histogram.  Need to add a pi shifted
%version of the histogram to itself before blurring.
bins2 = circshift(bins, [0 -200]); %there should be no overlap, I think?
% bins2(315) = 0;
all_quadrant_bins = bins + bins2;

%blur the histogram
blur = fspecial('gaussian', [1 15], 3);

% figure(3)
% plot(bins)

blurred_bins = conv(all_quadrant_bins, blur);
blurred_bins = blurred_bins(1,ceil(size(blur,2)/2):size(blurred_bins,2) - floor(size(blur,2)/2));

% figure(4)
% plot(blurred_bins)

%bin 157 (pi/2) equals bin 471 (3pi/2), so everything outside that range is redundant
blurred_bins(1:100) = 0;
blurred_bins(301:400) = 0;

% figure(4)
% polar(t, blurred_bins)
% figure(5)
% plot(t, blurred_bins)

angle_hist = blurred_bins(101:300); %the non zero elements, 200 elements.

% normalization
%for single global segment
angle_hist = angle_hist ./ area_normalization;
angle_hist = angle_hist ./ valid_percentage; %normalize for masked area.
% figure(5)
% plot(angle_hist)


%confidence is a scalar array with as many rows as lines.  
%segmentation is a segmentation image.  Features will be computed for each
%area of the image, and concatenated together for the final output.

%possibility 1 - global, simple statistics
%histogram of edge occurance by length. (weight bins corresponding to
%longer edges more) (weight by confidence)

length_hist = zeros(1,30);
for i = 1:size(lengths,1)
    current_length = lengths(i);
    length_bin = round(sqrt(current_length));
    if(length_bin > 30)
        length_bin = 30;
    end
    %min length is probably 10 or so?
    length_hist(length_bin) = length_hist(length_bin) + current_length * sqrt(confidences(i));
end

blur = fspecial('gaussian', [1 7], 1.5);
length_hist = conv(length_hist, blur);
length_hist = length_hist(1,ceil(size(blur,2)/2):size(length_hist,2) - floor(size(blur,2)/2));

length_hist = length_hist./area_normalization;
length_hist = length_hist./valid_percentage; %normalize for masked area.
% figure(6)
% plot(length_hist)
% pause

% angle_sum = sum(sum(angle_hist))
% length_sum = sum(sum(length_hist))

angle_hist = single(angle_hist);
length_hist = single(length_hist);
