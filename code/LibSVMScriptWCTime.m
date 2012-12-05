clear all;
close all;

%#if 0, %disable the info function...
% =============================
%     Step0. Declare Params
% =============================    
WORDS_LIMIT = 2001;
WORDS_STEP = 1000;
NUM_TOPICS =[50 75 100]; 
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
vocabFile = 'vocabulary.txt';             % list of words
featFile = 'feature.txt';                 % list of features sorted by patient_id
wcFile = 'patient_data.txt';              % sparse representation of wc
timeWCFile = 'patient_data_temporal.txt'; % sparse representation of wc in order
patientMatchFile = 'patient_rows.txt'; 


% =============================
%     Step1. Data Input
% =============================
[labels, features, icuStay, hospStay, survivalTime,  ...
   hospExpireFlag, readmit, totalHospStays, sid] = importCSVData(fullfile(dataPath, featFile));    
[numPat, numFeat] = size(features);

save featureFiles2.mat labels features icuStay hospStay ...
    survivalTime hospExpireFlag readmit totalHospStays; 

if PAT_SAVED == 1
    load 'patient_groups.mat';
else    
    %--------------------------
    % Randomly pick the training and evaluation sets
    %--------------------------          
    [train, test] = crossvalind('HoldOut', hospExpireFlag, .3);
    fprintf(1, [num2str(sum(train)) ' patients used to train topics models, ' ...
                num2str(sum(test)) 'patients used to create SVMs\n']);

    %--------------------------
    % Chi square tests
    %--------------------------
    fprintf(1, 'Chi Square/RankSum Tests for\nFeature\tTest(SVM)\tTrain(TopicModels)\n');
    groups = train;
    for i = 1:numFeat
        %binary
        if sum(features(:, i) == 0 | features(:, i) == 1) == length(features(:, i))
            n1 = sum(features(groups ==0, i)); N1 = length(features(groups == 0, i)); 
            n2 = sum(features(groups ==1, i)); N2 = length(features(groups == 1, i));

            x1 = [repmat('a',N1,1); repmat('b',N2,1)];
            x2 = [ones(n1,1); repmat(2,N1-n1,1); ones(n2,1); repmat(2,N2-n2,1)];
            [tbl, H, P] = crosstab(x1, x2);

            fprintf(1, [labels{i} '\t' num2str(nanmean(features(groups == 0, i)), '%.2f') ' ' 177 ' ' num2str(nanstd(features(groups == 0, i)), '%.2f') ... 
                    '\t' num2str(nanmean(features(groups == 1, i)), '%.2f') ' ' 177 ' ' num2str(nanstd(features(groups == 1, i)), '%.2f') ...
                    '\t' num2str(P, '%.5f') '\n']);    

        else    
            [P, H] = ranksum(features(groups == 0 & ~isnan(features(:, i)), i), features(groups == 1 & ~isnan(features(:, i)), i));

            fprintf(1, [labels{i} ...
                    '\t' num2str(nanmedian(features(groups == 0, i)), '%.2f') ' IQR ' num2str(iqr(features(groups == 0, i)), '%.2f') ...
                    '\t' num2str(nanmedian(features(groups == 1, i)), '%.2f') ' IQR ' num2str(iqr(features(groups == 1, i)), '%.2f') ...
                    '\t' num2str(P, '%.5f') '\n']);         
        end   

        clear x1 x2;
    end
    
    save 'patient_groups.mat' train test;
end

clear labels features icuStay hospStay ...
   hospExpireFlag readmit totalHospStays;

% =============================
%     Step2 Data Pre-processing
% =============================

if WC_SAVED == 1
    load TopicParams3.mat;
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
    scanStr = '%s%s%s%s%s%s%s%s%s%s%s%s%s';
    text = textscan(fid, scanStr, 'delimiter', '\t');
    fclose(fid);

    sid2 = cellfun(@str2num, text{1});
    dischargeNote = strcmp('DISCHARGE_SUMMARY', text{10});
    hoursFromFirstNote = cellfun(@str2num, text{13});

    figure;
    hist(hoursFromFirstNote/(24*365), 10);
    xlabel('Years')
    ylabel('Frequency');
    title('Note Time Distributions - All');
    close all;
    
    clear input;

    % Find the training rows, and the associated subj-IDs
    trainNumbers = find(train == 1);
    inds = ismember(sid2, sid(trainNumbers));

    wcTrain = wc(inds, :); trainIDs = unique(sid2(inds)); sidTrain = sid2(inds); hoursFromFirstNoteTrain = hoursFromFirstNote(inds);
    wcTest = wc(~inds, :); testIDs = unique(sid2(~inds)); sidTest = sid2(~inds); hoursFromFirstNoteTest = hoursFromFirstNote(~inds);
    clear wc hoursFromFirstNote;
    
    train(find(ismember(sid, setdiff(sid(train), trainIDs))))= 0;
    test(find(ismember(sid, setdiff(sid(test), testIDs))))= 0;
    save patient_groups.mat train test;
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
    train_TFIDF = bsxfun(@rdivide, wcTrainSum, sum(wcTrainSum,2)); %divide each row by the total number of words seen for that patient row1.all./sum(row1.all)
    train_TFIDF = bsxfun(@times, train_TFIDF, log(size( wcTrainSum , 1 )./(1+sum(wcTrainSum>0, 1)))); %create an IDF matrix by multiplying each column by 
    t1 = train_TFIDF(1:floor(size(train_TFIDF, 1)/4), :); 
    t2 = train_TFIDF(floor(size(train_TFIDF, 1)/4)+1:floor(size(train_TFIDF, 1)/2), :);
    t3 = train_TFIDF(floor(size(train_TFIDF, 1)/2)+1:floor(3*size(train_TFIDF, 1)/4), :); 
    t4 = train_TFIDF(floor(3*size(train_TFIDF, 1)/4)+1:end, :);
    clear train_TFIDF;
    %eval('pack;');

    [mat, wInd1] = sort(t1, 2, 'descend'); [mat, wInd2] = sort(t2, 2, 'descend');
    [mat, wInd3] = sort(t3, 2, 'descend'); [mat, wInd4] = sort(t4, 2, 'descend');
    clear mat t1 t2 t3 t4;
    keep = unique([unique(wInd1(:, 1:500)); ...
                   unique(wInd2(:, 1:500)); ...
                   unique(wInd3(:, 1:500));
                   unique(wInd4(:, 1:500))]); 
    clear wInd1 wInd2 wInd3 wInd4;
    removeTrain = setdiff(1:numWords, keep);

    % Remove the unneeded words from the test tfidf and wc matrix
    wcTrainSum(:, removeTrain) = [];
    wcTest(:, removeTrain) = [];
    WO(removeTrain) = [];
    fprintf( 'Check 3\n');

    % Test IDF is calculated from train documents
    test_TFIDF = bsxfun(@rdivide, wcTest, sum(wcTest,2)); %divide each row by the total number of words seen for that patient row1.all./sum(row1.all)
    test_TFIDF = bsxfun(@times, test_TFIDF, log(size( wcTrainSum , 1 )./(1+sum(wcTrainSum>0, 1)))); %create an IDF matrix by multiplying each column by 

    % Convert to sparse counts format and set toopic modelling constraints
    [ WS , DS ] = SparseMatrixtoCounts( wcTrainSum' );
    BETA = 200 / size(WS, 2);
    numPatients = size(wcTrainSum, 1);
    numWords = size( wcTrainSum, 2 );
    died = (survivalTime(train) == 0);
    long = (survivalTime(train) >= 1005);
    live = ~(died | long );
    lived1 = (survivalTime(train) > 0 & survivalTime(train) <= 30);
    lived6 = ~(died | long | lived1);
    
    
    save pcoriTopicParams.mat; numPatients numWords BETA sid2;
    save GibbsFile3.mat WS DS WO died long live; 
    save wcFiles3.mat wcTest test_TFIDF wcTrainSum -v7.3; 
    clear wcTrainSum WS DS WO died long lived;
    clear trainNumbers testOnlyWords emptyTestNotes ...
            dischargeOnlyWords dischargeNote inds removeTrain;
    clear wcTest test_TFIDF;
end

fprintf(1, 'Done with Word counting\n');

% =============================
%     Step3 Topic Modelling
% =============================        
matlabpool open local 4

dataStoreWP = []; %number of times word i assigned to topics j, size is (NUM_WORDSxNUM_TOPICS)
dataStoreDP = []; %number of times any word in doc i assigned to topics j, size is (NUM_PATIENTSxNUM_TOPICS)
wpKey = [];
dpKey = [];

eval('pack');

if TOPIC_SAVED == 1
    load 'ldasingle_WP3.mat';
else          
     parfor t = 1:length(NUM_TOPICS)
        numT = NUM_TOPICS(t);        
        ALPHA = 50/numT;

        S = load('GibbsFile3.mat');
        
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
               
        topics50 = bsxfun(@rdivide, DP50, 1+sum(DP50, 2)); 
%         L = WriteTopics( WP, BETA , S.WO, 10);
        %WriteData(WP, DP, Z, numT);
              
        %Normalize each row to sum to one
        topicsOfDead = sum(topics50(died, :))./sum(died);
        topicsOfLong = sum(topics50(long, :))./sum(long);
        topicsOfLive = sum(topics50(lived, :))./sum(lived);
%         topicsOfLive1 = sum(topics50(lived1, :))./sum(lived1);
%         topicsOfLive6 = sum(topics50(lived6, :))./sum(lived6);
%         
        topicsOfNotDead = sum(topics50(~died, :))./sum(~died);
        topicsOfNotLong = sum(topics50(~long, :))./sum(~long);
        topicsOfNotLive = sum(topics50(~lived, :))./sum(~lived);
%         topicsOfNotLive1 = sum(topics50(~lived1, :))./sum(~lived1);     
%         topicsOfNotLive6 = sum(topics50(~lived6, :))./sum(~lived6); 
%         
%         [val, ind] = sort(topicsOfDead./topicsOfNotDead, 'descend');
        
                        
        find(topicsOfDead./topicsOfNotDead & topicsOfDead > 0.02)
        find(topicsOfDead./topicsOfNotDead & topicsOfDead > 0.02)
        
        %plot images
        figure;
        %b = bar(topicsOfDead, 'r');
        %hold on; 
        %bar(topicsOfLive1, 'm');
        %bar(topicsOfLive, 'y');
        %bar(topicsOfLong, 'g'); 
        
        %bar([topicsOfDead./sum(died); topicsOfLive./sum(lived); topicsOfLong./sum(long)]', 'grouped');
        bar([median(topics50(died, :))*1000; ...
             median(topics50(lived, :))*1000; ...
             median(topics50(long, :))*1000]', 'grouped');
        
         
        
        topics50 = bsxfun(@rdivide, DP50, 1+sum(DP50, 2));
        topics50 = [mean(topics50(died, :)); mean(topics50(lived, :)); mean(topics50(long, :))];
        %topics50 = bsxfun(@rdivide, topics50, 1+sum(topics50, 1));
         
        bar([topics50(1, :); topics50(2, :); topics50(3, :)]', 'grouped');
        colormap([176/255 23/255 31/255; 255/255 193/255 37/255; 0 205/255 0]);
        xlabel('Topic ID');
        ylabel('Topic Dominance by Group');
%        title('Topic Membership (Stacked)');
        legend('In-Hospital Mortalities', 'Near-Term Mortalities', 'Long-Term Survival', 'Location', 'NorthEast');    
%         legend('In-Hospital Mortality', '0-30 Day Mortality', '1-6 Month Mortality', '> 6 Month Survival', 'Location', 'NorthEast');    
        xlim([0 51]);
%         %set(gcf, 'Position', [200 100 1000 650]);
%         %I = getframe(gcf);
%         %imwrite(I.cdata, ['visual_' num2str(numT) '.png']);
%         %close all;
        
        %WriteTopics( WP, BETA , WO , 10 , 1.0 , 4 , ['topTenWordsPerTopic_' num2str(numT) '.txt']);            
        %eval(['save ''ldasingle_topics.mat'' DP' num2str(numT)  WP' num2str(numT)  Z' num2str(numT) ' ''-append'';']);  
        %eval(['save ''ldasingle_WP.mat'' WP' num2str(numT) ' ''-append'';']);                                        
        
    end % for topics  
    
    save topicStruct3.mat dataStoreWP dataStoreDP wpKey dpKey;
end %if SKIP         

matlabpool close

fprintf(1, 'Done with Topic Modelling\n');

%Here we are splitting apart the datastructure which was generated as a
%result of rdeailing with matlab's peculiar form of parallelism
load topicStruct3.mat;
WO = load('GibbsFile3.mat', 'WO');
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
save 'ldasingle_topics3.mat' DP50 WP50 S50 DP75 WP75 S75 DP100 WP100 S100; 
save 'ldasingle_WP3.mat' WP50 WP75 WP100;

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


  