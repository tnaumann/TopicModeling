[label, feat] = libsvmread('/scratch/lib/libsvm-3.11/heart_scale');
folds = 5;

c_begin = -5; c_end = 15; c_step = 2;
g_begin = 3; g_end = -15; g_step = -2;
log2c = c_begin:c_step:c_end;
log2g = g_begin:g_step:g_end;

for i=1:length(log2c)*length(log2g)    
    c_iter = ceil(i / length(log2g));
    g_iter = mod(i - 1, length(log2g)) + 1;
    fprintf('%d\t%d (%d)\t%d (%d)\n', i, c_iter, log2c(c_iter), g_iter, log2g(g_iter));
    
    cvacc = svmtrain2(label, feat, ['-v ' num2str(folds) ' -c ', num2str(2^log2c(c_iter)), ' -g ', num2str(2^log2g(g_iter))]);
    
    thisParams = [cvacc log2c(c_iter) log2g(g_iter)];
    bestParams = comparemax(bestParams, thisParams);
end




[heart_scale_label, heart_scale_inst] = libsvmread('/scratch/lib/libsvm-3.11/heart_scale');
heart_scale_label = heart_scale_label + 1;
indices = crossvalind('Kfold', heart_scale_label, 5);
cp = classperf(heart_scale_label);



load featureFiles.mat; 
load patient_groups.mat;
double(features(train, 1:4)), double(hospExpireFlag(train))

indices = crossvalind('Kfold', hospExpireFlag, 5);

%using MATLAB SVM
for i = 1:5
   test = (indices == i); train = ~test;
   
   tic
   %SVMStruct = svmtrain(heart_scale_inst(train, :), heart_scale_label(train), 'kernel_function', 'rbf'); labels = svmclassify(SVMStruct, heart_scale_inst(test, :));
   SVMStruct = svmtrain(wc(train, :), hospExpireFlag(train), 'kernel_function', 'rbf');
   labels = svmclassify(SVMStruct, wc(test, :));
   toc
   classperf(cp, labels, test)
end

%using LIBSVM
for i = 1:5
   test = (indices == i); train = ~test;
   tic
   model = svmtrain2(hospExpireFlag(train), wc(train, :));%, '-c 1 -g 0.07');
   [predict_label, accuracy, dec_values] = svmpredict(hospExpireFlag(test), wc(test, :), model); 
   toc
   accuracy
   classperf(cp, labels, test);
end

%SVMStruct = svmtrain(heart_scale_inst, heart_scale_label, 'rbf_sigma', sigma, 'boxconstraint', C, 'kernel_function', 'rbf');
%cvacc = svmtrain2(heart_scale_label, heart_scale_inst, ['-v ' num2str(5) ' -c 1 -g 0.07']);