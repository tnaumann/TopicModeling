clear all;
load '/scratch/mghassem/modellingCode/dataForParForRun.mat';
clear featTemp; 

matlabpool open local 8
for i = 1:size(outcome, 2)
    i
    savedParamsWords(i, :) = parallelLibSVM2(i, log2c, log2g, folds, plotLabel);
end
matlabpool close