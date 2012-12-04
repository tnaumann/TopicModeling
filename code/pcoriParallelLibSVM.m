function [bestParams bestModel] = callLibSVM(label, feat, plotLabel, type)

% Make sure labels are in a column vector. 
label = double(label(:));

% Use a linear support vector machine classifier
folds = 5;
c_begin = -5; c_end = 15; c_step = 2;
g_begin = 3; g_end = -15; g_step = -2;

log2c = c_begin:c_step:c_end;
log2g = g_begin:g_step:g_end;  
indices = crossvalind('Kfold', label, folds);

Z = [];

%accuracy, c param, g param
bestParams = [0 1 1 0 0];

%I TOOK OUT THE PAR
for iter = 1:length(log2c)*length(log2g)
    c_iter = ceil(iter / length(log2g));
    g_iter = mod(iter - 1, length(log2g)) + 1;

    tps = [];
    tns = [];
    fps = [];
    fns = [];
    acc = [];
    for i = 1:5
        test = (indices == i); 
        train = ~test;

        % If single class
        if type == 2
            model = svmtrain2(label(train & label==1), feat(train & label==1, :), ['-s ' num2str(type) '  -h 0 -c ', num2str(2^log2c(c_iter)), ' -g ', num2str(2^log2g(g_iter))]);
        else
            model = svmtrain2(label(train), feat(train, :), ['-s ' num2str(type) '  -h 0 -c ', num2str(2^log2c(c_iter)), ' -g ', num2str(2^log2g(g_iter))]);
        end
        [predict_label, accuracy, dec_values] = svmpredict(label(test), feat(test, :), model); 
        %[~, ~, thresh, AUC, optPoint] = perfcurve(label(test), dec_values, 1);
        
        lab = label(test);
        [tpr, fpr, ~] = roc(lab(:)', dec_values(:)');
        auc = computeAUC(tpr,fpr);
        
        tp = sum(label(test) & predict_label);
        tn = sum(label(test)~=1 & predict_label~=1);
        fp = sum(label(test)~=1 & predict_label);
        fn = sum(label(test) & predict_label~=1);

        tps = [tps; tp];
        tns = [tns; tn];
        fps = [fps; fp];
        fns = [fns; fn];

        %acc = [acc; AUC]; %(tp+tn)/(tp+tn+fp+fn)
        
        sens = tp/(tp+fn); % also called recall.
        spec = tn/(tn+fp);

        ppv = tp/(tp+fp);  % also called precision
        npv = tn/(tn+fn);

        %acc = (tp+tn)/(tp+tn+fp+fn);

        fscore = 2*ppv*sens/(ppv+sens); % F1 score reaches its best value at 1 and worst score at 0.
        
        acc = [acc; auc];
    end

    cvacc = mean(acc); cvtp = mean(tps); cvtn = mean(tns); cvfp = mean(fps); cvfn = mean(fns);
    thisParams = [cvacc log2c(c_iter) log2g(g_iter) cvtp cvtn cvfp cvfn];
    
    % Custom reduction function for 3-element vector input
    if (cvacc > bestParams(1)) || ...
            (cvacc == bestParams(1)) && (log2c(c_iter) < bestParams(2)) && (log2g(g_iter) <= bestParams(3))
        bestParams = thisParams;
        bestModel = model;
    end

    Z = [Z; [c_iter, g_iter, thisParams(1)]];
end                                    

Z = spconvert(Z);

