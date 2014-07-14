function all_perf =compute_kernel_svm(para,conf)

all_perf = [];

outName = get_kernel_filename(para);

disp(outName);

%setup;

load(conf.splitFile);
split=split{para.splitID};
num_classes = length(split);

cnt = 0;
for i=1:num_classes
    for j=1:min(conf.max_num_train,length(split{i}.Training))
        cnt = cnt+1;
        class_train(cnt) = i;
    end
end

cnt = 0;
for i=1:num_classes
    for j=1:min(conf.max_num_test,length(split{i}.Testing))
        cnt = cnt+1;
        class_test(cnt)  =i;
    end
end


%%
all_changing_num_train = [1000 800 400 200 100 50 20 10 5 1];

changing_num_train = [];
first_in = true;
for k=1:length(all_changing_num_train)
    if all_changing_num_train(k) < conf.max_num_train
        if first_in
            changing_num_train = [conf.max_num_train];
            first_in = false;
        end
        changing_num_train = [changing_num_train, all_changing_num_train(k)];
    end
end

%% result exist?
cnt = 0;
for k=1:length(changing_num_train)
    num_train = changing_num_train(k);
    if exist([conf.resultPath 'SVM_Result_' sprintf('%.4d',num_train) outName],'file')
        cnt = cnt+1;
        load([conf.resultPath 'SVM_Result_' sprintf('%.4d',num_train) outName]);        
        C = confusionMatrix(class_test,class_hat');
        disp(sprintf('#train = %.4d   Performance = %f %%',num_train,mean(diag(C))));
        all_perf(k) = mean(diag(C));        
    end
end
if (cnt==length(changing_num_train)) % all exist
    disp('has been finished before');  
    return;
end


if exist(fullfile(conf.kernelPath, ['Test' outName]),'file')
    load(fullfile(conf.kernelPath, ['Train' outName]));
    load(fullfile(conf.kernelPath, ['Test' outName]));
else
    %% load training features
    Ftrain = cell(num_classes*conf.max_num_train,1);
    cnt = 0;
    for i=1:num_classes
        for j=1:min(conf.max_num_train,length(split{i}.Training))
            if conf.imageDir
                feat_fname = fullfile(conf.featPath,para.featureName,[split{i}.ClassName '/' split{i}.Training{j}(1:end-4) '.mat']);
            else
                feat_fname = fullfile(conf.featPath,para.featureName,['/' split{i}.Training{j}(1:end-4) '.mat']);
            end
            try
                cnt = cnt+1;
                Ftrain(cnt) = {load(feat_fname)};
                % Ftrain{cnt}.geom_c_map =[];
                % Ftrain{cnt}.texton_map =[];
            catch err
                disp(['feature cannot be loaded: ' feat_fname]);
            end
        end
    end
    disp('train features are loaded');
    
    %% pack training features X = |||||||
    X = packF(Ftrain,para.featureName,para.weighted,para.bow,para.normalize);
    X = single(X);
    clear Ftrain;
    disp('train features are packed');
    
    %% load testing features
    cnt = 0;
    for i=1:num_classes
        for j=1:min(conf.max_num_test,length(split{i}.Testing))
            if conf.imageDir
                feat_fname = fullfile(conf.featPath,para.featureName,[split{i}.ClassName '/' split{i}.Testing{j}(1:end-4) '.mat']);
            else
                feat_fname = fullfile(conf.featPath,para.featureName,['/' split{i}.Testing{j}(1:end-4) '.mat']);
            end
            try
                cnt = cnt+1;
                Ftest(cnt)  = {load(feat_fname)};
                % Ftest{cnt}.geom_c_map =[];
                % Ftest{cnt}.texton_map =[];
            catch err
                disp(['feature cannot be loaded: ' feat_fname]);
            end
        end
    end
    disp('test features are loaded');

    %% pack testing features
    Y = packF(Ftest,para.featureName,para.weighted,para.bow,para.normalize);
    Y = single(Y);
    clear Ftest;
    disp('test features are packed');
    
    if (~exist(fullfile(conf.kernelPath, ['Train' outName]),'file') || (strcmp(para.kernelName,'echi2')==1))
        % compute training and testing kernels
        [K K_test]= kernel(X,Y,para.kernelName,true);
        save('-v7.3',fullfile(conf.kernelPath, ['Train' outName]), 'K');        
        save('-v7.3',fullfile(conf.kernelPath, ['Test' outName]), 'K_test');
        disp('kernels are computed');
    else
        % compute testing kernels
        [K K_test] = kernel(X,Y,para.kernelName,false);
        % load training features
        if isempty(K)
            load(fullfile(conf.kernelPath, ['Train' outName]));
        end
        save('-v7.3',fullfile(conf.kernelPath, ['Test' outName]), 'K_test');
        disp('kernels are computed');
    end
    
end

%% train svm

for k=1:length(changing_num_train)
    clear selVec;
    num_train = changing_num_train(k);
    
    if k==1
        selVec = [1:length(class_train)];
    else
        cnt = 0;
        for i=1:num_classes
            for j=1:num_train
                cnt = cnt + 1;
                selVec(cnt) = (i-1)*changing_num_train(k-1)+j;
            end
        end
    end
    
    K = K(selVec,selVec);
    K_test = K_test(selVec,:);
    class_train = class_train(selVec);
    
    if ~exist([conf.resultPath 'SVM_Result_' sprintf('%.4d',num_train) outName],'file')
        score_test = svm_one_vs_all(K,K_test,class_train,num_classes);        
        % evaluation multi-class
        [confidence,class_hat] = max(score_test, [], 2);
        C = confusionMatrix(class_test,class_hat');
        disp(sprintf('#train = %.4d   Performance = %f %%',num_train,mean(diag(C))));
        all_perf(k) = mean(diag(C));
        save([conf.resultPath 'SVM_Result_' sprintf('%.4d',num_train) outName],'class_hat','score_test','confidence');
    end
end
