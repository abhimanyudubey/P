function combine_kernel(parameterCell,conf,paraCombine)


for i=1:length(parameterCell)
    parameterCell{i}.splitID = paraCombine.splitID;
end

for i=1:length(parameterCell)
    mu(i) = 1;
    
    disp(['Train ' num2str(i)]);
    outName = get_kernel_filename(parameterCell{i});
    
    load(fullfile(conf.kernelPath, ['Train' outName]));
    load(fullfile(conf.kernelPath, ['Test' outName]));
    
    if paraCombine.normalize
        mu(i) = mean(mean(K));
    end
    if i==1
        K_comb = parameterCell{i}.weight * (K/mu(i));
    else
        K_comb = K_comb + parameterCell{i}.weight * (K/mu(i));
    end
    
    disp(['Test ' num2str(i)]);
    load(fullfile(conf.kernelPath, ['Test' outName]));
    if i==1
        K_test_comb = parameterCell{i}.weight * (K_test/mu(i));
    else
        K_test_comb = K_test_comb + parameterCell{i}.weight * (K_test/mu(i));
    end
end
K=K_comb;
save(fullfile(conf.kernelPath, ['Train' get_kernel_filename(paraCombine)]),'-v7.3','K');
K_test=K_test_comb;
save(fullfile(conf.kernelPath, ['Test' get_kernel_filename(paraCombine)]),'-v7.3','K_test');

