%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Video LabelMe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

HOMEVIDEOS = 'http://labelme.csail.mit.edu/VideoLabelMe/VLMFrames/';
HOMEVIDEOANNOTATIONS = 'http://labelme.csail.mit.edu/VideoLabelMe/VLMAnnotations/';

HOMEVIDEOS = '/afs/csail.mit.edu/group/vision/www/data/LabelMe/VideoLabelMe/VLMFrames/';
HOMEVIDEOANNOTATIONS = '/afs/csail.mit.edu/group/vision/www/data/LabelMe/VideoLabelMe/VLMAnnotations/';



folder = folderlist(HOMEVIDEOANNOTATIONS, 'videos_iccv09');

D = LMdatabase(HOMEVIDEOANNOTATIONS, folder);
skipRate = 1; % change this if you want to play the video faster

for i = 1:length(D)
    LMplayfromframes(D, i, HOMEVIDEOS, skipRate);
end


