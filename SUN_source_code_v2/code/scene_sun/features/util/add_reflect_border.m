function [bordered_img] = add_reflect_border(img, border_pix)

%jhhays 4/22/2009
%gabor gist has strong border artifacts from the unpadded FFT2

if(size(img, 3) > 1)
    fprintf('Error, can only pad grayscale images at the moment\n')
    return
end

if(~exist('border_pix', 'var'))
    border_pix = round(size(img) * .2);
end

if(length(border_pix) < 2)
    border_pix = [border_pix border_pix];
end

%bordered_img = zeros( size(img) + border_pix * 2);

new_h = size(img,1) + border_pix(1) * 2;
new_w = size(img,2) + border_pix(2) * 2;

bordered_img(border_pix(1)+1:border_pix(1)+size(img,1), ...
             border_pix(2)+1:border_pix(2)+size(img,2)) = img;
         
% figure
% imshow(bordered_img)
% pause
% try
%now there's 8 regions to fill. 
%top / bottom will use flipud image.
%left / right will use fliplr image.
%four corners will use images flipped both ways?
%we can build it in four steps, actually.
%top
bordered_img(1:border_pix(1), :) = ...
    flipud(bordered_img(border_pix(1)+1:border_pix(1)*2,:));
bordered_img(size(img,1)+border_pix(1):new_h, :) = ...
    flipud(bordered_img( size(img,1):size(img,1) + border_pix(1),:));
%   flipud(bordered_img( size(img,1) - border_pix(1) + border_pix(1):size(img,1) + border_pix(1),:));
% figure
% imshow(bordered_img)
% pause

bordered_img(:, 1:border_pix(2)) = ...
    fliplr(bordered_img(:, border_pix(2)+1 : border_pix(2)*2));

bordered_img(:, size(img,2)+border_pix(2):new_w) = ...
    fliplr(bordered_img( :, size(img,2) : size(img,2) + border_pix(2)));
%   fliplr(bordered_img( :, size(img,2) + border_pix(2)- border_pix(2): size(img,2) + border_pix(2)));
% catch
%     disp('error');
% end
% figure
% imshow(bordered_img)
% pause



