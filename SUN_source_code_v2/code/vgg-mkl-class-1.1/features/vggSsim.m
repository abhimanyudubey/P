function feat = vggSsim(conf, im)
% OLDSSIM Legagy SSIM feature code
%
%  [FRAMES, DESCS] = OLDSSYM(CONF, IM)
%
%  Author:: Andrea Vedaldi

conf_ = struct ;
conf = override(conf_, conf) ;

if size(im, 3) > 1, im = rgb2gray(im) ; end

imagePath = sprintf('%s%d.jpg', tempname, vl_getpid()) ;
[imageDir,imageName,imageExt] = fileparts(imagePath) ;
imwrite(im, imagePath) ;

[features, ftrCoords, salientCoords, uniformCoords] = getSSimFeatures(imagePath, conf) ;

delete(imagePath) ;

feat.frames = ftrCoords ;
feat.descrs = features ;
