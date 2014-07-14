function [lines, confidences, edgeIm] = APPgetLargeConnectedEdges_fast2(grayIm, minLen, input_mask)
%
% [lines, edgeIm] = APPgetLargeConnectedEdges(grayIm, minLen, input_mask)
% 
% Uses the method of Video Compass [Kosecka, et al 2002] to get long,
% straight edges.
% 
% Input:
%   grayIm: grayscale image to be analyzed
%   minLen: minimum length in pixels for edge (suggested 0.025*diagonal)
% Output:
%   lines: parameters for long, straight lines
%   confidences: the confidence value for each line.
%   edgeIm (optional): for displaying lines found
%
% Copyright(C) Derek Hoiem, Carnegie Mellon University, 2005
% Permission granted to non-commercial enterprises for
% modification/redistribution under GNU GPL.  
% Current Version: 1.0  09/30/2005
%
% this function has been heavily re-written by James Hays, incorporating
% calls to open cv to make it 5 to 10 times faster than the
% original function.

if(~exist('input_mask', 'var'))
    input_mask = zeros(size(grayIm));
end

% t2 = clock;
blurred_img = imfilter(grayIm,  fspecial('gaussian', 5, 1.5), 'same');
% elapsed_time = etime(clock,t2);
% fprintf('... took %f seconds\n', elapsed_time);

% t2 = clock;
% [dX, dY] = gradient(double(blurred_img));
% elapsed_time = etime(clock,t2);
% fprintf('... took %f seconds\n', elapsed_time);

% t2 = clock;
dX = cvlib_mex('filter', single(blurred_img), [-1 0 1; -2 0 2; -1 0 1]);
dY = cvlib_mex('filter', single(blurred_img), [-1 -2 -1;0 0 0; 1 2 1]);
% elapsed_time = etime(clock,t2);
% fprintf('... took %f seconds\n', elapsed_time);

% figure(3)
% imagesc(dX)
% figure(4)
% imagesc(dY)

% [im_canny, thresh] = edge(grayIm, 'canny');
%need to blur int_img, since matlab canny is doing it and it seems to help.

gauss_filt = fspecial('gaussian', 5, 1);
int_img = imfilter(grayIm, gauss_filt, 'same');
% [im_canny] = cvlib_mex('canny', int_img, [200 500]);
im_canny = cvlib_mex('canny',int_img, 200, 500, 5);

im_canny = im_canny .* (1 - input_mask);

% remove border edges
im_canny([1 2 end-1 end], :) = 0;
im_canny(:, [1 2 end-1 end]) = 0;
width = size(im_canny, 2);
height = size(im_canny, 1);

% figure(1)
% imshow(im_canny)

ind = find(im_canny > 0);

num_dir = 8;

dX_ind = dX(ind);
dY_ind = dY(ind);
a_ind = atan(dY_ind ./ (dX_ind+1E-10));

% a_ind ranges from 1 to num_dir with bin centered around pi/2
a_ind = ceil(mod(a_ind/pi*num_dir-0.5, num_dir));
%[g, gn] = grp2idx(a_ind);

% get the indices of edges in each direction
for i = 1:num_dir
    direction(i).ind = ind(find(a_ind==i));
end

% remove edges that are too small and give all edges that have the same
% direction a unique id
% edges(height, width, [angle id])

lines       = zeros(2000, 6);
confidences = zeros(2000,1);

% nspdata = length(imsegs.npixels);
% spdata = repmat(struct('lines', zeros(5, 1), 'edge_count', 0), nspdata,1);
% bcount = zeros(nspdata, 1);

used = zeros(size(im_canny));

line_count = 0;
for k = 1:num_dir
    
    num_ind = 0;
    for m = (k-1):k+1
        num_ind = num_ind + sum(~used(direction(mod(m-1, num_dir)+1).ind));
    end

    ind = zeros(num_ind, 1);
    dir_im = zeros(size(im_canny));
  
    count = 0;
    for m = (k-1):k+1
        m2 = mod(m-1, num_dir)+1;
        tind = direction(m2).ind(find(~used(direction(m2).ind)));
        tmpcount = length(tind);
        ind(count+1:count+tmpcount) = tind;
        count = count + tmpcount;
    end  
    dir_im(ind) = 1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % these are CONTOURS, not edge segments, so they double back
    % which is kind of weird, but I think it should work out.
    B = cvlib_mex('contours',logical(dir_im));
    num_edges = size(B,1);
    
%     [cont_image] = plot_contours(dir_im, B);
%     figure(2)
%     imagesc(cont_image)
    
       
    % get the endpoints of the long edges and an image of the long edges
    for id = 1:num_edges

        edge_size = size(B{id},1) * .52;  %why is this so slow??
        
        if edge_size > minLen
            
            current_contour = B{id};

            y = current_contour(:,1);
            x = current_contour(:,2);
            
            mean_x = mean(x);
            mean_y = mean(y);
            zmx = (x-mean_x);
            zmy = (y-mean_y);
            D = [sum(zmx.^2) sum(zmx.*zmy); sum(zmx.*zmy) sum(zmy.^2)];
            [v, lambda] = eig(D);
%             D
%             v
%             lambda
            %why atan2?  shouldn't there be a pi ambiguity??
            %theta = atan2(v(2, 2) , v(1, 2));
            
            %this is not the normal to the line, this is the angle of the
            %line.
            if(v(2,2) ~= 0)
                theta = atan( v(1, 2) / v(2, 2)) - pi/2;
            else
                theta = pi;
            end
%             theta
            
            if lambda(1,1)>0
                conf = lambda(2,2)/lambda(1,1);
            else
                conf = 1600;
            end
                
            %disp(conf)
            
            if conf >= 200 
                line_count = line_count+1;
                
%                 used(edges(id).ind) = 1;
                used( sub2ind(size(dir_im), current_contour(:,1),current_contour(:,2))) = 1;
%                 imshow(used)
%                 pause

                %disp(num2str([lambda(1,1) lambda(2,2)]))
                r = sqrt((max(x)-min(x))^2 + (max(y)-min(y))^2);
                %this is length.
                
                %x is affected by sin because theta is not the angle of the
                %line, but an angle perpendicular to the line.
                x1 = mean_x + cos(theta)*r/2;
                x2 = mean_x - cos(theta)*r/2;
                y1 = mean_y - sin(theta)*r/2;
                y2 = mean_y + sin(theta)*r/2;            
                
                r = mean_x*cos(theta)+mean_y*sin(theta);
                %this is not length, this is distance from the origin
                            
                lines(line_count, 1:6) = [x1 x2 y1 y2 theta r];
                confidences(line_count,1) = conf;
	
            end
        end
    end
    
end

% t0 = clock;
% elapsed_time = etime(clock,t0);
% fprintf('Main loop took %f seconds\n', elapsed_time);

% if nargout>2
% %     edgeIm = grayIm*0.75;
%     edgeIm = zeros(size(grayIm));
% 
%     for i = 1:line_count
%         edgeIm = draw_line_image2(edgeIm, lines(i, 1:4)', confidences(i)/800,  i);
%     end
% end

% for k = 1:length(spdata)
%     spdata(k).lines = spdata(k).lines(1:bcount(k))';
% end

lines = lines(1:line_count, :);
confidences = confidences(1:line_count,:);
% imsize = size(grayIm);
% lines = normalizeLines(lines, imsize(1:2));
