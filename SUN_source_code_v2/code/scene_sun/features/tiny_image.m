function [feat boxes] = tiny_image(conf,current_img)

    if(size(current_img,3) < 3)
        %black and white image
        current_img = cat(3, current_img, current_img, current_img); %make it a trivial color image
    end
    
    tiny_image = single(imresize(current_img, [32 32]));
    tiny_image = tiny_image(:);
    feat.tiny_image = tiny_image;
    boxes = [1;1;32;32];