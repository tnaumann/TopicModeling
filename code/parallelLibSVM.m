function [bestParams] = parallelLibSVM(label, feat, log2c, log2g, folds, plotLabel)

    for i = 1:size(feat, 2) 
        if sum(isnan(feat(:, i))) > 0
            feat(isnan(feat(:, i)), i) = nanmean(feat(:, i));
        end
    end
    indices = crossvalind('Kfold', label, folds);

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
        for i = 1:5
            test = (indices == i); 
            train = ~test;
                   
            %%currently using C-SVC versus nu-SVC... may want to change and
            %%add -w0 1 -w1 5 to modify class imbalcnces; may also want to
            %%use -b 1 to get probablity estimates for classes
            %model = svmtrain2(label(train), feat(train, :), ['-m 500 -h 0 -w0 5 -w1 10 -c ', num2str(2^log2c(c_iter)), ' -g ', num2str(2^log2g(g_iter))]);
            %%[tmp index]=ismember(model.SVs, training_data,'rows');
            %[predict_label, accuracy, dec_values] = svmpredict(label(test), feat(test, :), model); 
            
            model = svmtrain2(label(train), feat(train, :), ['-h 0 -c ', num2str(2^log2c(c_iter)), ' -g ', num2str(2^log2g(g_iter))]);
            [predict_label, accuracy, dec_values] = svmpredict(label(test), feat(test, :), model); 
            [X,Y,T,AUC,OPTROCPT] = perfcurve(label(test), dec_values, 0);
            
            tp = sum(label(test) & predict_label);
            tn = sum(~label(test) & ~predict_label);
            fp = sum(~label(test) & predict_label);
            fn = sum(label(test) & ~predict_label);
            
            %sens = [sens; tp/(tp+fn)];
            %spec = [spec; tn/(tn+fp)];
            tps = [tps; tp];
            tns = [tns; tn];
            fps = [fps; fp];
            fns = [fns; fn];
            
            acc = [acc; AUC]; %(tp+tn)/(tp+tn+fp+fn)
        end
        
        cvacc = mean(acc); cvtp = mean(tps); cvtn = mean(tns); cvfp = mean(fps); cvfn = mean(fns);
        
        thisParams = [cvacc log2c(c_iter) log2g(g_iter) cvtp cvtn cvfp cvfn];
        bestParams = comparemax(bestParams, thisParams);

        Z = [Z; [c_iter, g_iter, thisParams(1)]];
    end                                    

    %savedParamsTopic(k, :) = bestParams;
%    Z = spconvert(Z);
%     figure;
%     plotBullsEye(Z, bestParams(1), bestParams(2), bestParams(3), bestParams(4), bestParams(5), ...
%        log2c(1), log2c(end), log2g(1), log2g(end));
%     set(gcf, 'Position', [500 1000 1000 600]);
%     I = getframe(gcf);
%     imwrite(I.cdata, [plotLabel '.png']);
%    close all;     
    
    
% %     -Parameters: parameters
% %         -nr_class: number of classes; = 2 for regression/one-class svm
% %         -totalSV: total #SV
% %         -rho: -b of the decision function(s) wx+b
% %         -Label: label of each class; empty for regression/one-class SVM
% %         -ProbA: pairwise probability information; empty if -b 0 or in one-class SVM
% %         -ProbB: pairwise probability information; empty if -b 0 or in one-class SVM
% %         -nSV: number of SVs for each class; empty for regression/one-class SVM
% %         -sv_coef: coefficients for SVs in decision functions
% %         -SVs: support vectors
