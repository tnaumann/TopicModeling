function [bestParams] = parallelLibSVM2(index, log2c, log2g, folds, plotLabel)    
    Z = [];

    %accuracy, c param, g param
    bestParams = [0 1 1 0 0];
    
    parfor iter = 1:length(log2c)*length(log2g)
        c_iter = ceil(iter / length(log2g));
        g_iter = mod(iter - 1, length(log2g)) + 1;
	
        fprintf('%d\t%d (%d)\t%d (%d)\n', iter, c_iter, log2c(c_iter), g_iter, log2g(g_iter)); 
        %cvacc = svmtrain2(label, feat, ['-v ' num2str(folds) ' -c ', num2str(2^log2c(c_iter)), ' -g ', num2str(2^log2g(g_iter))]);        
        
        tps = [];
        tns = [];
        fps = [];
        fns = [];
        acc = [];
        
        S = load('dataForParForRun.mat');
        label = double(S.outcome(:, index));        
        
        indices = crossvalind('Kfold', label, folds);
        
        for i = 1:5
            test = (indices == i); 
            train = ~test;
                   
            model = svmtrain2(label(train), double(S.featTemp(train, :)), ['-h 0 -c ', num2str(2^log2c(c_iter)), ' -g ', num2str(2^log2g(g_iter))]);
            [predict_label, accuracy, dec_values] = svmpredict(label(test), double(S.featTemp(test, :)), model); 
            
            tp = sum(label(test) & predict_label);
            tn = sum(~label(test) & ~predict_label);
            fp = sum(~label(test) & predict_label);
            fn = sum(label(test) & ~predict_label);
            
            tps = [tps; tp];
            tns = [tns; tn];
            fps = [fps; fp];
            fns = [fns; fn];
            
            acc = [acc; accuracy(1)]; %(tp+tn)/(tp+tn+fp+fn)
        end
        
        cvacc = mean(acc); cvtp = mean(tps); cvtn = mean(tns); cvfp = mean(fps); cvfn = mean(fns);
        
        thisParams = [cvacc log2c(c_iter) log2g(g_iter) cvtp cvtn cvfp cvfn];
        bestParams = comparemax(bestParams, thisParams);

        Z = [Z; [c_iter, g_iter, thisParams(1)]];
    end                                    

    %savedParamsTopic(k, :) = bestParams;
    Z = spconvert(Z);
    %figure;
    %plotBullsEye(Z, bestParams(1), bestParams(2), bestParams(3), bestParams(4), bestParams(5), ...
    %   log2c(1), log2c(end), log2g(1), log2g(end));
    %set(gcf, 'Position', [500 1000 1000 600]);
    %I = getframe(gcf);
    %imwrite(I.cdata, [plotLabel '.png']);
    %close all;               