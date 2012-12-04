function auc=computeAUC(tpr,fpr)


%%%%%
% This function computes AUC given true and false positive rates, using the
% trapezoidal rule. Use the "roc" function in Matlab to get the input
% variables tpr and fpr.
%
% Inputs:
% tpr - 1D vector of true positive rates
% fpr - 1D vector of false positive rates, should correspond to the elements in tpr
%
% Outputs:
% auc - area under the ROC curve
%
% Written by Joon Lee, 2011
%%%%%


tpr=tpr(:);
fpr=fpr(:);
[fpr,IX]=sort(fpr);
tpr=tpr(IX);

% make sure the first point is (0,0)
if tpr(1)~=0 && fpr(1)~=0
    tpr=[0; tpr];
    fpr=[0; fpr];
end

% make sure the last point is (1,1)
if tpr(end)~=1 && fpr(end)~=1
    tpr=[tpr; 1];
    fpr=[fpr; 1];
end

auc=sum(((tpr(1:end-1)+tpr(2:end))./2).*diff(fpr));
