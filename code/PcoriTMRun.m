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

TOPIC_SAVED = 1;
WC_SAVED = 1;
PAT_SAVED = 1;
BASE_SAVED = 0;

% Directory Paths to Data
dataPath = '/home/mghassem/Documents/MATLAB/studies/pcori/';
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
    outcome = daysToDeathFromFirstNote < 30;
    [train, test] = crossvalind('HoldOut', outcome, .3);
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
    save pcoriGibbsFile.mat WS DS WO died live updateD2D; 
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
        topicsOfDead = sum(topics50(died, :))./sum(died);
        topicsOfLive = sum(topics50(live, :))./sum(live);
        topicsOfNotDead = sum(topics50(~died, :))./sum(~died);
        topicsOfNotLive = sum(topics50(~live, :))./sum(~live);
         
        [val, ind] = sort(topicsOfDead./topicsOfNotDead, 'descend')
                               
        find(topicsOfDead./topicsOfNotDead & topicsOfDead > 0.02)
        
        %plot images
        figure;                
        bar([topicsOfDead; ...
             topicsOfLive]', 'grouped');
         colormap([176/255 23/255 31/255; 0 205/255 0]);
        xlabel('Topic ID');
        ylabel('Median Topic Membership by Group');
        legend('30-Day Mortalities', '30-Day Survival', 'Location', 'NorthEast');   
        xlim([0 51]);
%         
%         %plot images
%         figure;                
%         bar([median(topics50(died, :))*1000; ...
%              median(topics50(live, :))*1000]', 'grouped');
%         xlabel('Topic ID');
%         ylabel('Median Topic Membership by Group');
%         legend('30-Day Mortalities', '30-Day Survival', 'Location', 'NorthEast');    
%         
%         
        figure;
        topics50 = bsxfun(@rdivide, DP50, 1+sum(DP50, 2));                        
        meansPerTopic = mean(topics50, 1); stdPerTopic = std(topics50, 1);
        negThresh = meansPerTopic - stdPerTopic;
        posThresh = meansPerTopic + stdPerTopic;
        
        posInd = (topics50 > repmat(posThresh, numPatients, 1));
        negInd = (topics50 < repmat(negThresh, numPatients, 1));
        topics50(:, :) = 0;
        topics50(negInd) = -1;
        topics50(posInd) = 1;
        
        topics50 = [mean(topics50(died, :)); mean(topics50(live, :))];         
        bar([topics50(1, :); topics50(2, :)]', 'grouped');
        colormap([176/255 23/255 31/255; 0 205/255 0]);
        xlabel('Topic ID');
        ylabel('Topic Dominance (based on 1/#topics threshold) by Group');
        legend('30-Day Mortalities', '30-Day Survival', 'Location', 'NorthEast');    
        xlim([0 51]);
        
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
DP20 = dataStoreDP((numPatients*(find(dpKey == 1) - 1) + 1):(numPatients*(find(dpKey == 1))), 1:20);
WP20 = dataStoreWP((numWords*(find(wpKey == 1) - 1) + 1):(numWords*(find(wpKey == 1))), 1:20);
topics20 = bsxfun(@rdivide, DP20, 1+sum(DP20, 2)); 
S20 = WriteTopics(WP20, BETA , WO, 20);%, 1.0, 20, 'topTenWordsPerTopic_20.txt');

DP35 = dataStoreDP((numPatients*(find(dpKey == 2) - 1) + 1):(numPatients*(find(dpKey == 2))), 1:35); 
WP35 = dataStoreWP((numWords*(find(wpKey == 2) - 1) + 1):(numWords*(find(wpKey == 2))), 1:35); 
topics35 = bsxfun(@rdivide, DP35, 1+sum(DP35, 2)); 
S35 = WriteTopics(WP35, BETA , WO, 20);%, 1.0, 20, 'topTenWordsPerTopic_35.txt');

DP50 = dataStoreDP((numPatients*(find(dpKey == 3) - 1) + 1):(numPatients*(find(dpKey == 3))), 1:50);
WP50 = dataStoreWP((numWords*(find(wpKey == 3) - 1) + 1):(numWords*(find(wpKey == 3))), 1:50);
topics50 = bsxfun(@rdivide, DP50, 1+sum(DP50, 2)); 
S50 = WriteTopics(WP50, BETA , WO, 20);%, 1.0, 20, 'topTenWordsPerTopic_50.txt');

%DP matrices are the Document( pateitn) x toic counts, 
%Wp are the wordxtopic counts
%topic matrices are the normalized distrinbutions per patient
%S matrices are the top 10 owrds for each number of topics
save 'pcoriLDASingle_topics.mat' DP20 WP20 S20 DP35 WP35 S35 DP50 WP50 S50; 
save 'pcoriLDASingle_WP.mat' WP20 WP35 WP50;

% =============================
%     Step4. SVM Train/Test on WCSUm, this is per patient
% =============================  
load pcoriTopicParams.mat; % numPatients numWords BETA sid2;
load pcoriWCFiles.mat;     % wcTest test_TFIDF wcTrainSum -v7.3; 
load pcoriGibbsFile.mat;

%--------------------------
% Outcome: survivalTime > 30
%--------------------------
outcome = double(updateD2D < 30);
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

savedParamsTopic = zeros(length(NUM_TOPICS), 7);
savedParamsWords = zeros(length(NUM_WORDS), 5);
savedPerfTopic = zeros(length(NUM_TOPICS), 6);
savedPerfWords = zeros(length(NUM_WORDS), 6);

% remove the 0 day mortalities
allInd =  updateD2D > 0;
plotLabel = 'bullsEyeBaseline';

%--------------------------
%Model with topics only
%--------------------------
for k = 1:length(NUM_TOPICS)        
    
    % Get train population's topic membership    
    eval(['normfeat = topics' num2str(NUM_TOPICS(k)) ';']);            
    plotLabel = ['bullsEye' num2str(NUM_TOPICS(k)) 'Topics'];
    [bestParams, bestModel] = pcoriParallelLibSVM(outcome(train), normfeat, 'two_class', 0);    

    % Evaluate on the test set
    eval(['topicMem = wcTestSum*WP' num2str(NUM_TOPICS(k)) ';']);
    topicMem = bsxfun(@rdivide, topicMem, 1+sum(topicMem, 2));
    [predict_label1, accuracy1, dec_values1] = ...
            svmpredict(double(outcome(test)'), double(topicMem), bestModel);    
    [sens, spec, ppv, npv, acc, fscore] = ...
            summaryOfPerf(double(outcome(test)), double(dec_values1), [num2str(k) ': Best two class SVM Model on 20% held out data had ']);

    savedParamsTopic(k, :) = bestParams;
    savedPerfTopic(k, :) = [sens spec ppv npv acc fscore];
end

% Plot for the test set
died = (updateD2D(test) < 30);
live = updateD2D(test) >= 30;
topicsOfDead = sum(topicMem(died, :))./sum(died);
topicsOfLive = sum(topicMem(live, :))./sum(live);
topicsOfNotDead = sum(topicMem(~died, :))./sum(~died);
topicsOfNotLive = sum(topicMem(~live, :))./sum(~live);

%plot images
figure;                
bar([topicsOfDead; ...
     topicsOfLive]', 'grouped');
 colormap([176/255 23/255 31/255; 0 205/255 0]);
xlabel('Topic ID');
ylabel('Median Topic Membership by Group');
legend('30-Day Mortalities', '30-Day Survival', 'Location', 'NorthEast');   
xlim([0 51]);

figure;
meansPerTopic = mean(topicMem, 1); stdPerTopic = std(topicMem, 1);
negThresh = meansPerTopic - stdPerTopic;
posThresh = meansPerTopic + stdPerTopic;

posInd = (topicMem > repmat(posThresh, 300, 1));
negInd = (topicMem < repmat(negThresh, 300, 1));
topics50 = zeros(size(topicMem));
topics50(negInd) = -1;
topics50(posInd) = 1;

topics50 = [mean(topics50(died, :)); mean(topics50(live, :))];         
bar([topics50(1, :); topics50(2, :)]', 'grouped');
colormap([176/255 23/255 31/255; 0 205/255 0]);
xlabel('Topic ID');
ylabel('Topic Dominance (based on 1/#topics threshold) by Group');
legend('30-Day Mortalities', '30-Day Survival', 'Location', 'NorthEast');    
xlim([0 51]);

fprintf(1, 'Done with Topic SVM Outcomes\n');
save pcoriAllVars3.mat savedParamsBase savedParamsBaseEH ...
     savedParamsBaseTopic savedParamsBaseWords -append;

% Print them 
savedParamsTopic

%--------------------------
% Pick only the top N words
%--------------------------
tfidfSum = zeros(length(testIDs), size(test_TFIDF, 2));
tfidfSum = S*test_TFIDF;
[perPatientNoteNum, id] = hist(sidTest, unique(sidTest));
tfidfSum = tfidfSum ./ repmat(perPatientNoteNum', 1, size(tfidfSum, 2));

[~, wInd] = sort(test_TFIDF, 2, 'descend'); clear mat t1; 
wInd1 = wInd1(:, 1:5000);  

%--------------------------
% Word models
%--------------------------
outcome = outcome(test, :);
for i=1:length(NUM_WORDS)
    numWords = NUM_WORDS(i);  
    keep = unique(wInd1(:, 1:numWords));

    featTemp = tfidfSum(:, keep);
    featTemp(:, find(sum(featTemp) == 0)) = [];

    %scale the feature matrix
    featTemp = (featTemp - repmat(min(featTemp,[],1), ...
                       size(featTemp,1),1)) ...
               *spdiags(1./(max(featTemp,[],1) ...
                        -min(featTemp,[],1))',0, ...
                        size(featTemp,2), ...
                        size(featTemp,2));   %     
    save dataForParForRun.mat outcome test featTemp log2c log2g folds plotLabel -v7.3;
    clear featTemp;

        %--- Words model with only words
        plotLabel = ['bullsEye' num2str(numWords) 'Words'];
        %savedParamsWords(j, i, :) = parallelLibSVM2(j, log2c, log2g, folds, plotLabel);
        savedParamsWords(i, :) = parallelLibSVM(double(outcome(test, j)), double(featTemp), log2c, log2g, folds, plotLabel);
end

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
  
