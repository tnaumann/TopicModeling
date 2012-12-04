function [sens, spec, ppv, npv, acc, fscore] = summaryOfPerf(singleClassLabels, dec_values, label)

singleClassLabels = singleClassLabels(:);
dec_values = dec_values(:);
predict_label = zeros(size(singleClassLabels));

[fpr, tpr, thresh, AUC, OPTROCPT] = perfcurve(singleClassLabels, dec_values, 1);

[fpr, accu, thresh] = perfcurve(singleClassLabels, dec_values, 1,'ycrit','accu');
% plot(thre,accu);
% xlabel('Threshold for ''good'' Returns');
% ylabel('Classification Accuracy');
[maxaccu,iaccu] = max(accu);
optimalthresh = thresh(iaccu);
predict_label(dec_values < optimalthresh) = -1;
predict_label(dec_values >= optimalthresh) = 1;

tp = sum(singleClassLabels & predict_label);
tn = sum(singleClassLabels==-1 & predict_label==-1);
fp = sum(singleClassLabels==-1 & predict_label);
fn = sum(singleClassLabels & predict_label==-1);
sens = tp/(tp+fn); % also called recall.
spec = tn/(tn+fp);

ppv = tp/(tp+fp);  % also called precision
npv = tn/(tn+fn);

acc = (tp+tn)/(tp+tn+fp+fn);

fscore = 2*ppv*sens/(ppv+sens); % F1 score reaches its best value at 1 and worst score at 0.

fprintf(1, '%s had an AUC %0.3f and accuracy of %0.3f (Sens %0.3f, Spec %0.3f, PPV %0.3f, NPV %0.3f, FScore %0.3f)\n', ...
            label, AUC, acc, sens, spec, ppv, npv, fscore);
