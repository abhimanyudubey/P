function im = getImage(varargin)
% GETIMAGE Read an image
%  IM = GETIMAGE(IMAGEPATH)
  if length(varargin) > 1
    imagePath = fullfile(varargin{:}) ;
  else
    imagePath = varargin{1} ;
  end
  if exist(imagePath, 'file')
  elseif exist([imagePath '.jpg'], 'file')
    imagePath = [imagePath '.jpg'] ;
  else
    error('Could not find image ''%s''', imagePath) ;
  end
  im = imread(imagePath) ;
end