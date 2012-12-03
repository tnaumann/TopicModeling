clear all;
close all;

%#if 0, %disable the info function...
% =============================
%     Step0. Declare Params
% =============================    
WORDS_LIMIT = 2001;
WORDS_STEP = 1000;
NUM_TOPICS =[20 35 50]; 
NUM_WORDS = [500 1000 2000];
MAX_TOPICS = 200;

N = 5000;
SEED = 1:length(NUM_TOPICS);
OUTPUT = 1;

TOPIC_SAVED = 0;
WC_SAVED = 0;
PAT_SAVED = 0;
BASE_SAVED = 0;

% Directory Paths to Data
dataPath = '/scratch/mghassem/pcori-pilot-packed';
vocabFile = 'vocabulary.txt';        % list of words
featFile = 'feature.txt';           % list of features sorted by patient_id
wcFile = 'patient_data.txt';         % sparse representation of wc
timeWCFile = 'patient_data_temporal.txt'; % sparse representation of wc in order
patientMatchFile = 'patient_rows.txt'; 

% RBF SVM Training Params
folds = 5;
c_begin = -5; c_end = 15; c_step = 2;
g_begin = 3; g_end = -15; g_step = -2;

log2c = c_begin:c_step:c_end;
log2g = g_begin:g_step:g_end;  

numSteps = WORDS_LIMIT/WORDS_STEP;


% =============================
%     Step1. Data Input
% =============================
fid = fopen(fullfile(dataPath, featFile));
text = textscan(fid, '%d');
fclose(fid);
daysToDeathFromFirstNote = double(text{1});
[numPat] = length(daysToDeathFromFirstNote);
sid = 1:numPat;

% =============================
%     Step2 Data Pre-processing
% =============================

if PAT_SAVED == 1
    load pcoriPatientGroups.mat;
else    
    %--------------------------
    % Randomly pick the training and evaluation sets
    %--------------------------          
    [train, test] = crossvalind('HoldOut', numPat, .3);
    fprintf(1, [num2str(sum(train)) ' patients used to train topics models, ' ...
                num2str(sum(test)) ' patients used to create SVMs\n']);
            
    save pcoriPatientGroups.mat train test;
end

if WC_SAVED == 1
    load pcoriTopicParams.mat;
else
    %--------------------------
    % Read in the total vocab list and remove words from it
    %--------------------------
    fid = fopen(fullfile(dataPath, vocabFile));
    WO = textscan(fid, '%s', 'Delimiter', '\n', 'Headerlines', 0); 
    WO = WO{1};
    fclose(fid);

    %--------------------------
    % Load the counts files so that we are only 
    % looking at patients from the training set
    %--------------------------   
    input = load(fullfile(dataPath, timeWCFile)); wc = spconvert(input);
    fprintf( 'Number of Documents D = %d\n' , size( wc , 1 ));
    fprintf( 'Number of Words     W = %d\n' , size( wc , 2 ));
    fprintf( 'Number of nonzero entries in matrix NNZ=%d\n' , nnz( wc ));

    %--------------------------
    % Load the data from the other file
    %--------------------------   
    fid = fopen(fullfile(dataPath, patientMatchFile));
    scanStr = '%s%s%s%s%s%s';
    text = textscan(fid, scanStr, 'delimiter', ' ');
    fclose(fid);

    sid2 = cellfun(@str2num, text{1});
    dischargeNote = strcmp('DIS', text{3});
    hoursFromFirstNote = cellfun(@str2num, text{6});

    figure;
    hist(hoursFromFirstNote/(24*365), 10);
    xlabel('Years')
    ylabel('Frequency');
    title('Note Time Distributions - All');

    % Modify daysToDeath to be from last noet
    updateD2D = daysToDeathFromFirstNote(sid2) - hoursFromFirstNote/24;    
    S = bsxfun(@eq, sid, sid2);
    perColD2D = S.*repmat(updateD2D, 1, numPat); perColD2D(perColD2D == 0) = NaN;
    updateD2D = min(perColD2D); updateD2D(isnan(updateD2D)) = 0;

    figure;
    hist(updateD2D/365, 30);
    xlabel('Years')
    ylabel('Frequency');
    title('Days from Last Note to Death Distributions - All');    
    
    % Find the training rows, and the associated subj-IDs
    trainNumbers = find(train == 1);
    inds = ismember(sid2, sid(trainNumbers));    

    wcTrain = wc(inds, :); trainIDs = unique(sid2(inds)); sidTrain = sid2(inds); hoursFromFirstNoteTrain = hoursFromFirstNote(inds);
    wcTest = wc(~inds, :); testIDs = unique(sid2(~inds)); sidTest = sid2(~inds); hoursFromFirstNoteTest = hoursFromFirstNote(~inds);
    clear wc hoursFromFirstNote;
    
    train(find(ismember(sid, setdiff(sid(train), trainIDs))))= 0;
    test(find(ismember(sid, setdiff(sid(test), testIDs))))= 0;
    save pcoriPatientGroups.mat train test;
    fprintf( 'Check 1\n');

    %remove words that are in the test set only from both
    testOnlyWords = (sum(wcTrain, 1) == 0);

    wcTest(:, testOnlyWords) = [];
    wcTrain(:, testOnlyWords) = [];
    WO(testOnlyWords) = [];

    %remove notes that are now empty from test
    emptyTestNotes = sum(wcTest, 2) == 0;
    %wcTest(emptyTestNotes, :) = [];

    %comglomerate the training files to per-patient, not per-note
    S = bsxfun(@eq, trainIDs, sidTrain');
    numWords = size(wcTrain, 2);
    wcTrainSum = zeros(length(trainIDs), numWords);
    wcTrainSum = S*wcTrain;    
    clear S wcTrain;
    fprintf( 'Check 2\n');

    %--------------------------
    % Calculate TF-IDF Scores
    %--------------------------

    % Filter to top 500 words by patient using the tfidf scores
    % Divide each row by the total number of words seen for that patient row1.all./sum(row1.all)
    train_TFIDF = bsxfun(@rdivide, wcTrainSum, sum(wcTrainSum,2)); 
    
    % Create an IDF matrix by multiplying each column by 
    train_TFIDF = bsxfun(@times, train_TFIDF, log(size( wcTrainSum , 1 )./(1+sum(wcTrainSum>0, 1)))); 
    [~, wInd] = sort(train_TFIDF, 2, 'descend');
    keep = unique(wInd(:, 1:500));
    clear wInd1;
    removeTrain = setdiff(1:numWords, keep);

    % Remove the unneeded words from the test tfidf and wc matrix
    wcTrainSum(:, removeTrain) = [];
    wcTest(:, removeTrain) = [];
    WO(removeTrain) = [];
    fprintf( 'Check 3\n');

    % Test IDF is calculated from train documents
    test_TFIDF = bsxfun(@rdivide, wcTest, sum(wcTest,2)); %divide each row by the total number of words seen for that patient row1.all./sum(row1.all)
    test_TFIDF = bsxfun(@times, test_TFIDF, log(size( wcTrainSum , 1 )./(1+sum(wcTrainSum>0, 1)))); %create an IDF matrix by multiplying each column by 

    % Convert to sparse counts format and set topic modelling constraints
    [ WS , DS ] = SparseMatrixtoCounts( wcTrainSum' );
    BETA = 200 / size(WS, 2);
    
    numPatients = size(wcTrainSum, 1);
    numWords = size( wcTrainSum, 2 );
    
    died = (updateD2D(train) < 30);
    live = updateD2D(train) >= 30;
    
    save pcoriTopicParams.mat numPatients numWords BETA sid2;
    save pcoriGibbsFile.mat WS DS WO died live; 
    save pcoriWCFiles.mat wcTest test_TFIDF wcTrainSum -v7.3; 
    clear wcTrainSum WS DS WO died live;
    clear trainNumbers testOnlyWords emptyTestNotes ...
            dischargeOnlyWords dischargeNote inds removeTrain;
    clear wcTest test_TFIDF;
end

fprintf(1, 'Done with Word counting\n');

% =============================
%     Step3 Topic Modelling
% =============================        
%matlabpool open local 4

dataStoreWP = []; %number of times word i assigned to topics j, size is (NUM_WORDSxNUM_TOPICS)
dataStoreDP = []; %number of times any word in doc i assigned to topics j, size is (NUM_PATIENTSxNUM_TOPICS)
wpKey = [];
dpKey = [];

eval('pack');

if TOPIC_SAVED == 1
    load 'pcoriLDASingle_WP.mat';
else 
    %parfor
     for t = 1:length(NUM_TOPICS)
        numT = NUM_TOPICS(t);        
        ALPHA = 50/numT;

        S = load('pcoriGibbsFile.mat');
        
        %Infer the topics and write them to a text file
        tic;
        [WP, DP, Z] = GibbsSamplerLDA( S.WS , S.DS , numT , N , ALPHA , BETA , SEED(t), OUTPUT );
        toc
        
        %save this thread's WP
        wpPad = zeros(numWords, MAX_TOPICS); wpPad(:, 1:numT) = WP;
        dataStoreWP = [wpPad; dataStoreWP];
        wpKey = [t; wpKey];

        %save this thread's DP
        dpPad = zeros(numPatients, MAX_TOPICS); dpPad(:, 1:numT) = DP;
        dataStoreDP = [dpPad; dataStoreDP];
        dpKey = [t; dpKey];
               
%         DP50 = DP;
%               
%         %Normalize each row to sum to one
%         topics50 = full(bsxfun(@rdivide, DP50, 1+sum(DP50, 2))); 
%         topicsOfDead = sum(topics50(died, :))./sum(died);
%         topicsOfLive = sum(topics50(live, :))./sum(live);
%         topicsOfNotDead = sum(topics50(~died, :))./sum(~died);
%         topicsOfNotLive = sum(topics50(~live, :))./sum(~live);
%          
%         [val, ind] = sort(topicsOfDead./topicsOfNotDead, 'descend')
%                                
%         find(topicsOfDead./topicsOfNotDead & topicsOfDead > 0.02)
%         
%         %plot images
%         figure;                
%         bar([topicsOfDead; ...
%              topicsOfLive]', 'grouped');
%          colormap([176/255 23/255 31/255; 0 205/255 0]);
%         xlabel('Topic ID');
%         ylabel('Median Topic Membership by Group');
%         legend('30-Day Mortalities', '30-Day Survival', 'Location', 'NorthEast');   
%         
%         %plot images
%         figure;                
%         bar([median(topics50(died, :))*1000; ...
%              median(topics50(live, :))*1000]', 'grouped');
%         xlabel('Topic ID');
%         ylabel('Median Topic Membership by Group');
%         legend('30-Day Mortalities', '30-Day Survival', 'Location', 'NorthEast');    
% 
%         topics50 = bsxfun(@rdivide, DP50, 1+sum(DP50, 2));                        
%         meansPerTopic = mean(topics50, 1); stdPerTopic = std(topics50, 1);
%         negThresh = meansPerTopic - stdPerTopic;
%         posThresh = meansPerTopic + stdPerTopic;
%         
%         posInd = (topics50 > repmat(posThresh, numPatients, 1));
%         negInd = (topics50 < repmat(negThresh, numPatients, 1));
%         topics50(:, :) = 0;
%         topics50(negInd) = -1;
%         topics50(posInd) = 1;
%         
%         
%         figure;
%         topics50 = [mean(topics50(died, :)); mean(topics50(live, :))];         
%         bar([topics50(1, :); topics50(2, :)]', 'grouped');
%         colormap([176/255 23/255 31/255; 0 205/255 0]);
%         xlabel('Topic ID');
%         ylabel('Topic Dominance (based on 1/#topics threshold) by Group');
%         legend('30-Day Mortalities', '30-Day Survival', 'Location', 'NorthEast');    
%         xlim([0 21]);
        
%         %set(gcf, 'Position', [200 100 1000 650]);
%         %I = getframe(gcf);
%         %imwrite(I.cdata, ['visual_' num2str(numT) '.png']);
%         %close all;
        
        %WriteTopics( WP, BETA , WO , 10 , 1.0 , 4 , ['topTenWordsPerTopic_' num2str(numT) '.txt']);            
        %eval(['save ''ldasingle_topics.mat'' DP' num2str(numT)  WP' num2str(numT)  Z' num2str(numT) ' ''-append'';']);  
        %eval(['save ''ldasingle_WP.mat'' WP' num2str(numT) ' ''-append'';']);                                        
        
    end % for topics  
    
    save pcoriTopicStruct.mat dataStoreWP dataStoreDP wpKey dpKey;
end %if SKIP         

%matlabpool close

fprintf(1, 'Done with Topic Modelling\n');

%Here we are splitting apart the datastructure which was generated as a
%result of rdeailing with matlab's peculiar form of parallelism
load pcoriTopicStruct.mat;
WO = load('pcoriGibbsFile.mat', 'WO');
WO = WO.WO;
find(dpKey == 1)
DP50 = dataStoreDP((numPatients*(find(dpKey == 1) - 1) + 1):(numPatients*(find(dpKey == 1))), 1:50);
WP50 = dataStoreWP((numWords*(find(wpKey == 1) - 1) + 1):(numWords*(find(wpKey == 1))), 1:50);
topics50 = bsxfun(@rdivide, DP50, 1+sum(DP50, 2)); 
S50 = WriteTopics(WP50, BETA , WO, 10);

DP75 = dataStoreDP((numPatients*(find(dpKey == 2) - 1) + 1):(numPatients*(find(dpKey == 2))), 1:75); 
WP75 = dataStoreWP((numWords*(find(wpKey == 2) - 1) + 1):(numWords*(find(wpKey == 2))), 1:75); 
topics75 = bsxfun(@rdivide, DP75, 1+sum(DP75, 2)); 
S75 = WriteTopics(WP75, BETA , WO, 10);

DP100 = dataStoreDP((numPatients*(find(dpKey == 3) - 1) + 1):(numPatients*(find(dpKey == 3))), 1:100);
WP100 = dataStoreWP((numWords*(find(wpKey == 3) - 1) + 1):(numWords*(find(wpKey == 3))), 1:100);
topics100 = bsxfun(@rdivide, DP100, 1+sum(DP100, 2)); 
S100 = WriteTopics(WP100, BETA , WO, 10);

%DP matrices are the Document( pateitn) x toic counts, 
%Wp are the wordxtopic counts
%topic matrices are the normalized distrinbutions per patient
%S matrices are the top 10 owrds for each number of topics
save 'pcoriLDASingle_topics.mat' DP50 WP50 S50 DP75 WP75 S75 DP100 WP100 S100; 
save 'pcoriLDASingle_WP.mat' WP50 WP75 WP100;

% =============================
%     Step4. SVM Train/Test on WCSUm, this is per patient
% =============================  
load featureFiles2.mat;
load wcFiles3.mat;

%--------------------------
% Outcomes are: 
% 1) hospExpireFlag
% 2) survivalTime > 30
% 3) survivalTime > 30*6
% 4) readmit < 30*6
%--------------------------
%outcome(hospExpireFlag) = 0;
%outcome(survivalTime > 30) = 1;
%outcome(survivalTime > 30*6) = 2;
%outcome = outcome';
outcome = [hospExpireFlag(test) survivalTime(test) < 30 survivalTime(test) < 30*6]; %readmit < 30*6];
testNumbers = find(test == 1);
inds = ismember(sid2, sid(testNumbers));
testIDs = unique(sid2(inds)); sidTest = sid2(inds);

%--------------------------
% Comglomerate the test files to per-patient, not per-note
%--------------------------
S = bsxfun(@eq, testIDs, sidTest');
numWords = size(wcTest, 2);
wcTestSum = zeros(length(testIDs), numWords);
wcTestSum = S*wcTest;

%--------------------------
% Create baseline models with ages, sex, SAPS and SOFA, and those plus EHs
%--------------------------
feat = features(test, :);
feat = (feat - repmat(min(feat,[],1), ...
                   size(feat,1),1)) ...
           *spdiags(1./(max(feat,[],1) ...
                    -min(feat,[],1))',0, ...
                    size(feat,2), ...
                    size(feat,2));   %

%[cvacc log2c(c_iter) log2g(g_iter) cvtp cvtn cvfp cvfn]
savedParamsBase = zeros(3, 5);
savedParamsBaseEH = zeros(3, 5);   
savedParamsTopic = zeros(3, length(NUM_TOPICS), 7);
savedParamsBaseTopic = zeros(3, length(NUM_TOPICS), 7);
savedParamsWords = zeros(3, length(NUM_WORDS), 5);
savedParamsBaseWords = zeros(3, length(NUM_WORDS), 5);
load allVars3.mat;

matlabpool open local 8

allInd = logical(ones(size(feat, 1), 1));
for j = 1:size(outcome, 2)

    %after in hosp mort, remove all the ones that die in hosp
    if j > 1
        allInd = ~hospExpireFlag(test);
    end
    
    plotLabel = 'bullsEyeBaseline';
    savedParamsBase(j, :) = parallelLibSVM(double(outcome(allInd, j)), ...
                        double(feat(allInd, 1:6)), log2c, log2g, folds, plotLabel);
    plotLabel = 'bullsEyeBaselineEH';
    savedParamsBaseEH(j, :) = parallelLibSVM(double(outcome(allInd, j)),  ...
                       double(feat(allInd, :)), log2c, log2g, folds, plotLabel);
end

fprintf(1, 'Done with Baseline SVM Outcomes\n');
save allVars3.mat savedParamsBase savedParamsBaseEH;

%--------------------------
%Model with topics only
%--------------------------
for k = 1:length(NUM_TOPICS)
    allInd = logical(ones(size(feat, 1), 1));
    for j = 1:size(outcome, 2)
        
        %after in hosp mort, remove all the ones that die in hosp
        if j > 1
            allInd = ~hospExpireFlag(test);
        end
        
        % Get test population's topic membership    
        eval(['topicMem = wcTestSum*WP' num2str(NUM_TOPICS(k)) ';']);
        normfeat = bsxfun(@rdivide, topicMem, 1+sum(topicMem, 2));

        %TODO iterative refinment of c/g
        %use only topics as features
        plotLabel = ['bullsEye' num2str(NUM_TOPICS(k)) 'Topics'];
        savedParamsTopic(j, k, :) = parallelLibSVM(double(outcome(allInd, j)), double(normfeat(allInd, :)), log2c, log2g, folds, plotLabel);

        %add the topics as features, run the libSVM and save output
        %featTemp = [feat(:, 1:6) normfeat];
        %plotLabel = ['bullsEye' num2str(NUM_TOPICS(k)) 'BaseTopics'];
        %savedParamsBaseTopic(j, k, :) = parallelLibSVM(double(outcome(test & allInd, j)), double(featTemp(allInd, :)), log2c, log2g, folds, plotLabel);
    end
end

fprintf(1, 'Done with Topic SVM Outcomes\n');
save allVars3.mat savedParamsBase savedParamsBaseEH ...
     savedParamsBaseTopic savedParamsBaseWords -append;
 
 
%--------------------------
% Pick only the top N words
%--------------------------
tfidfSum = zeros(length(testIDs), size(test_TFIDF, 2));
tfidfSum = S*test_TFIDF;
[perPatientNoteNum, id] = hist(sidTest, unique(sidTest));
tfidfSum = tfidfSum ./ repmat(perPatientNoteNum', 1, size(tfidfSum, 2));

firstQ = floor(size(test_TFIDF, 1)/4);
secondQ = floor(size(test_TFIDF, 1)/2);
thirdQ = floor(3*size(test_TFIDF, 1)/4);

t1 = test_TFIDF(1:firstQ, :); 
t2 = test_TFIDF(firstQ+1:secondQ, :);
t3 = test_TFIDF(secondQ+1:thirdQ, :); 
t4 = test_TFIDF(thirdQ+1:end, :);
clear test_TFIDF;
eval('pack;');

[mat, wInd1] = sort(t1, 2, 'descend'); clear mat t1; wInd1 = wInd1(:, 1:5000);  
[mat, wInd2] = sort(t2, 2, 'descend'); clear mat t2; wInd2 = wInd2(:, 1:5000);  
[mat, wInd3] = sort(t3, 2, 'descend'); clear mat t3; wInd3 = wInd3(:, 1:5000);  
[mat, wInd4] = sort(t4, 2, 'descend'); clear mat t4; wInd4 = wInd4(:, 1:5000); 

%--------------------------
% Word models
%--------------------------
outcome = outcome(test, :);
for i=1:length(NUM_WORDS)
    numWords = NUM_WORDS(i);  
    keep = unique(wInd1(:, 1:numWords));%; wInd2(:, 1:numWords); wInd3(:, 1:numWords); wInd4(:, 1:numWords)]);

    featTemp = tfidfSum(:, keep);
    featTemp(:, find(sum(featTemp) == 0)) = [];
    %featTemp(:, find(sum(featTemp) == 0)) = [];

    %scale the feature matrix
    featTemp = (featTemp - repmat(min(featTemp,[],1), ...
                       size(featTemp,1),1)) ...
               *spdiags(1./(max(featTemp,[],1) ...
                        -min(featTemp,[],1))',0, ...
                        size(featTemp,2), ...
                        size(featTemp,2));   %     
    save dataForParForRun.mat outcome test featTemp log2c log2g folds plotLabel -v7.3;
    clear featTemp;

	for j = 1:size(outcome, 2)-1
        %--- Words model with only words
        plotLabel = ['bullsEye' num2str(numWords) 'Words'];
        savedParamsWords(j, i, :) = parallelLibSVM2(j, log2c, log2g, folds, plotLabel);
        %savedParamsWords(j, i, :) = parallelLibSVM(double(outcome(test, j)), double(featTemp), log2c, log2g, folds, plotLabel);
        %featTemp = [feat(:, 1:4) featTemp];

        %--- Word Model without EH, 
        %plotLabel = ['bullsEye' num2str(numWords) 'BaseWords'];
        %savedParamsBaseWords(j, i, :) = parallelLibSVM(double(outcome(test, j)), double(featTemp), log2c, log2g, folds, plotLabel);
    end
end

matlabpool close

fprintf(1, 'Done with Word SVM Outcomes\n');

% =============================
%     Step5. Report SVM Performance
% =============================
save allVars3.mat savedParamsBase savedParamsBaseEH ...
     savedParamsBaseTopic savedParamsBaseWords ...
     savedParamsWords savedParamsTopic -append;
 
 outcomeLabels = {'In-hospital Expire', ' Survival Time > 30 days', ...
                  ' Survival Time > 6 months', ' Readmitted Within 6 months'};
 
for j = 1:size(outcome, 2) 
     figure;
     title({[outcomeLabels(j); ': SVM Performance under 5-fold Cross-Validation']});
     plot([1 1000 2000], ones(1, 3)*savedParamsBase(:, 1), 'b', 'Linewidth', 2);
     hold on;
     plot([1 1000 2000], savedParamsWords(j, :, 1), 'g', 'Linewidth', 2);
     plot([1 666 1333 2000], savedParamsTopic(j, :, 1), 'k', 'Linewidth', 2);
     xlabel('Number of Words');
     ylabel('Accuracy');
     legend('Base Model', 'Word Features', 'Topic Features');
     addTopXAxis('expression', 'argu.*.1', 'xLabStr', 'Number of Topics');
end

% =============================
%     Step6. SVM Train/Test on WCSUm, this is per note!
% =============================  


  