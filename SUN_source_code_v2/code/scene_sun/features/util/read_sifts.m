function [sift_params, sift_values] = read_sifts(sift_txt_path)
%James Hays, 9/23/2008

if(~exist('sift_txt_path', 'var'))
    sift_txt_path = 'out.desc';
end

fid = fopen(sift_txt_path);

num_dims  = str2num(fgetl(fid));
num_feats = str2num(fgetl(fid));

format = '%f32%f32%f32%f32%f32%f32%f32%f32%f32%f32%f32%f32%f32'; %the location , size, scale of features
for i = 1:num_dims %the dimensions of the sift
    format = [format '%u16'];
end

sifts = textscan(fid, format, num_feats, 'CollectOutput', 1);

sift_params = sifts{1};
sift_values = sifts{2};
sift_values(sift_values > 255) = 255; %not sure if this can happen
sift_values = uint8(sift_values);
fclose(fid);

% max(max(sift_values)) = 170
%  16 bit unsigned int would definitely be safe.