function outName = get_kernel_filename(para)

    outName = ['_' para.featureName ];
    if isfield(para,'splitID')
        outName = [outName '__split_' sprintf('%.2d',para.splitID)];
    end
    outName = [outName '__' para.kernelName '__'];
    if para.weighted
        outName = [outName 'weighted_' 'T' '__'];
    else
        outName = [outName 'weighted_' 'F' '__'];
    end
    if para.bow
        outName = [outName 'bow_' 'T' '__'];
    else
        outName = [outName 'bow_' 'F' '__'];
    end
    if para.normalize
        outName = [outName 'normalize_' 'T' '__'];
    else
        outName = [outName 'normalize_' 'F' '__'];
    end
    
    outName = [outName '.mat'];