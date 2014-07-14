function [feat boxes] = sparse_sift_hists(conf,current_img)

    if(size(current_img,3) < 3)
        %black and white image
        current_img = cat(3, current_img, current_img, current_img); %make it a trivial color image
    end
    boxes = [1;1;size(current_img,1);size(current_img,2)];

    % Sift features
    sift_input_image = rgb2gray(current_img);
    sift_input_image = rescale_max_size(sift_input_image, 500);
    [mser_sift_params, mser_sift_values, hesaff_sift_params, hesaff_sift_values] = calc_sift_features(sift_input_image);
    [hesaff_hist_flat] = assign_visual_words_flat(single(hesaff_sift_values), conf.hesaff_vocab_flat.cluster_centers1);
    [mser_hist_flat]   = assign_visual_words_flat(single(mser_sift_values),   conf.mser_vocab_flat.cluster_centers1);

    feat.hists.hesaff = hesaff_hist_flat;
    feat.hists.mser = mser_hist_flat;
