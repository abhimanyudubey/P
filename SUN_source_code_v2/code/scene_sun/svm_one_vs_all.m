function score_test = svm_one_vs_all(K,K_test,class_train,num_classes)

score_test = zeros(size(K_test,2), num_classes);
for class_ind = 1:num_classes
    %train an SVM for each class, test against all test cases.
    Y = 2*(class_train == class_ind)-1; %pos = 1, neg = -1
    
    %[beta,b]=primal_svm(0, Y, lambda);
    %score_test(:,class_ind) = K_test*beta+b;
    libsvm_cl = svmtrain(Y(:), double([(1:length(class_train))' K]), [' -t 4 -s 0 -w-1 1 -w1 ' num2str(length(find(Y==-1))/length(find(Y==1))) ' -c 1.0']) ;
    ap = mean(libsvm_cl.sv_coef(Y(libsvm_cl.SVs) > 0)) ;
    am = mean(libsvm_cl.sv_coef(Y(libsvm_cl.SVs) < 0)) ;
    if ap < am
        % fprintf('svmflip: SVM appears to be flipped. Adjusting.\n') ;
        libsvm_cl.sv_coef  = - libsvm_cl.sv_coef ;
        libsvm_cl.rho      = - libsvm_cl.rho ;
    end
    % test it on train
    train_scores = libsvm_cl.sv_coef' * K(libsvm_cl.SVs, :) - libsvm_cl.rho ;
    errs = train_scores .* Y < 0 ;
    err  = mean(errs) ;
    %fprintf('class %d: training error = %f\t # of SV=%d/%d\n', class_ind, err, length(libsvm_cl.SVs), length(Y));
    % test it on test
    score_test(:,class_ind) = libsvm_cl.sv_coef' * K_test(libsvm_cl.SVs,:) - libsvm_cl.rho ;
end
