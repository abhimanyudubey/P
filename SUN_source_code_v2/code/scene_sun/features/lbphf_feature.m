function [feat boxes] = lbphf_feature(conf, im)

if(size(im,3) < 3)
    %black and white image
    im = cat(3, im, im, im); %make it a trivial color image
end
boxes = [1;1;size(im,1);size(im,2)];


[lbphf_L0 lbphf_L1 lbphf_L2] = lbp_hf_4x4(im);

feat.hists.L0 = lbphf_L0;
feat.hists.L1 = lbphf_L1;
feat.hists.L2 = lbphf_L2;
