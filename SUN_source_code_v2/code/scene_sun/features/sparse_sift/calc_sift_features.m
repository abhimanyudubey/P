function [mser_sift_params, mser_sift_values, hesaff_sift_params, hesaff_sift_values] = calc_sift_features(image, vis_path)


%jhhays, 9/23/2008

%image_path is a path to a .jpg
%or an image

% file format 2:
% vector_dimension
% nb_of_descriptors
% x y cornerness scale/patch_size angle object_index  point_type laplacian_value extremum_type mi11 mi12 mi21 mi22 desc_1 ...... desc_vector_dimension
% --------------------
% 
% distance=(descA_1-descB_1)^2+...+(descA_vector_dimension-descB_vector_dimension)^2

% ./extract_features_64bit.ln -hesaff -hesThres 400 -sift -i London_00163_383132699_73d5.png -o2 out.desc -DE

%the last MSER point option might be able to feed into ./extract_features
%for sifts 

% ./mser.ln -i Italy_00100_453717251_4f9e5.png -t 2 -per .04
% ./extract_features_64bit.ln -sift -i Italy_00100_453717251_4f9e5.png -p1 Italy_00100_453717251_4f9e5.png.aff -DE

%detect MSER interest points
%create SIFT features at mser interest points

%create SIFT features at hessian-affine interest points

if(~exist('image', 'var'))
    image = 'test.jpg';
end


%if a visualization directory was passed in, draw all the sifts and save
%them there
vis = exist('vis_path', 'var');

if(ischar(image))  %if a path was passed in
    slashies = strfind(image, filesep);
    if(isempty(slashies))
        file_name = image;
    else
        file_name = image(max(slashies)+1:end);
    end
    file_name = file_name(1:end-4); %removing extension
else %it was an image that was passed in
    %rand('twister',sum(100*clock));
    %file_name = ['tmp_sift_img_' sprintf('%.11d', round(rand(1) * 100000000000))];
    
    rand('twister',sum(100*rand(1)*clock));
    [tfdamp hhname]=system('hostname');
    hhname = hhname(1:end-1);
    file_name = ['tmp_sift_img_' hhname sprintf('_%.0f', round(rand(1) * 100000000000))];    
end

% tmpdir = '/csail/vision-torralba/tmp/jhhays_sift/';
tmpdir = './';
if(~exist(tmpdir, 'file'))
    mkdir(tmpdir)
end
tmp_img_path = [tmpdir file_name '.png'];

if(ischar(image))  %if a path was passed in
    fprintf('downsampling and converting to .png\n')
    tmp_img = imread(image);
    tmp_img = double(tmp_img)/255;
else %it was an image that was passed in
    if(max(image) > 1)
        image = double(image)/255;
    end
    
    if(size(image,3) > 1)
        image = rgb2gray(image);
    end
    
    if(size(image,1) > 500 || size(image,2) > 500)
        image = rescale_max_size(image, 500);
    end
    
    tmp_img = image;
end

prefilt = 1;
if(prefilt) %this will normalize the local contrast, which will strongly affect the detection of interest points
    tmp_img = prefilt_jhhays(tmp_img);
end

imwrite( tmp_img, tmp_img_path );

fprintf('computing SIFT at MSER interest points\n')
% mser_command = ['calc_vid_goog/mser.ln -i ' tmp_img_path ' -t 2 -ms 60 -per .1 -mm 8 -o ' tmpdir file_name '.mser_int_points'];
mser_command = ['../sift_vgg/mser.ln -i ' tmp_img_path ' -t 2 -ms 60 -per .1 -mm 10 -o ' tmpdir file_name '.mser_int_points'];
system(mser_command);
sift_mser_command = ['../sift_vgg/extract_features_64bit.ln -sift -i ' tmp_img_path ' -p1 ' tmpdir file_name '.mser_int_points' ' -o2 ' tmpdir file_name '.msersifts.desc'];
if(vis)
    sift_mser_command = [sift_mser_command ' -DE'];
end
system(sift_mser_command);

fprintf('computing SIFT at hesaff interest points\n')
% sift_hesaff_command = ['calc_vid_goog/extract_features_64bit.ln -hesaff -hesThres 1200 -sift -i ' tmp_img_path ' -o2 ' tmpdir file_name '.hesaffsifts.desc'];
sift_hesaff_command = ['../sift_vgg/extract_features_64bit.ln -hesaff -hesThres 1600 -sift -i ' tmp_img_path ' -o2 ' tmpdir file_name '.hesaffsifts.desc'];
if(vis)
    sift_hesaff_command = [sift_hesaff_command ' -DE'];
end
system(sift_hesaff_command);


if(exist([tmpdir file_name '.msersifts.desc'], 'file'))
    [mser_sift_params,    mser_sift_values]  = read_sifts([tmpdir file_name '.msersifts.desc']);
    delete( [tmpdir file_name '.msersifts.desc'] )
else
    mser_sift_params = zeros(0,13,  'single');
    mser_sift_values = zeros(0,128, 'uint16');
end

if(exist([tmpdir file_name '.hesaffsifts.desc'], 'file'))
    [hesaff_sift_params, hesaff_sift_values] = read_sifts([tmpdir file_name '.hesaffsifts.desc']);
    delete( [tmpdir file_name '.hesaffsifts.desc'] )
else
    hesaff_sift_params = zeros(0,13,  'single');
    hesaff_sift_values = zeros(0,128, 'uint16');
end

if(vis)
    movefile( [tmpdir file_name '.msersifts.desc.png'], vis_path )
    movefile( [tmpdir file_name '.hesaffsifts.desc.png'], vis_path )
end

delete( [tmpdir file_name '.mser_int_points'] )
delete( [tmpdir file_name '.msersifts.desc.params'] )
delete( [tmpdir file_name '.hesaffsifts.desc.params'] )
delete( tmp_img_path );



