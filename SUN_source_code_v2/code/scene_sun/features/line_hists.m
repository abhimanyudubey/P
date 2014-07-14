function [feat boxes] = line_hists(conf,current_img)
    if(size(current_img,3) < 3)
        %black and white image
        current_img = cat(3, current_img, current_img, current_img); %make it a trivial color image
    end
    boxes = [1;1;size(current_img,1);size(current_img,2)];

    % Line features
    original_image_size = single(size(current_img));
    [lines, conf, spdata] = calc_lines_img(current_img);
    [angle_hist, length_hist] = calc_line_feats_img(lines, conf, zeros(original_image_size(1), original_image_size(2)), [], []);
    feat.hists.angle = angle_hist(:);
    feat.hists.length = length_hist(:);
