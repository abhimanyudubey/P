function extractFeature(imageName,conf)

for featName = conf.featNames
    clear feat
    clear hists
    clear temp
    featName = char(featName);
    feat_fname = fullfile(conf.featPath, featName, [imageName '.mat']);
    if ~exist(feat_fname,'file')
        
        if 0==exist('im','var')
            % load image
            try
                % important to use im2double
                im = im2double(imread(fullfile(conf.imagePath, [imageName '.jpg'])));
            catch
                disp(['ERROR: cannot open image ' imageName]);
                return;
            end
            
            % resize
            max_size = 640;
            h = size(im,1);
            w = size(im,2);
            scale_f = min(max_size/h,max_size/w);
            if scale_f < 1
                im = imresize(im, scale_f);
            end
        end
        
        tic
        % Extraction
        [feat boxes] = feval(conf.(featName).extractFn, conf.(featName), im) ;
        
        if conf.(featName).outside_quantization
            % compute hist
            temp = getRoiHistsFeat(feat.words, conf.(featName), conf.(featName).pyrLevels, boxes, 'vocabWeights', []) ;
            clear feat;

            % fix output format
            for li=1:length(conf.(featName).pyrLevels)
                histName = sprintf('L%d', conf.(featName).pyrLevels(li)) ;
                feat.hists.(histName) = temp{li};
            end
        end
        ssave(feat_fname, '-STRUCT', 'feat') ;
        fprintf('image %s    feature %s    time %f \n', imageName, featName, toc);
    end
end