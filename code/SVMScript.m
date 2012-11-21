clear all;
close all;

#if 0, %disable the info function...
% =============================
%     Step1. Data Scanning
% =============================
WORDS_LIMIT = 2000;
WORDS_STEP = 500;
NUM_TOPICS = 50:50:200; 
BETA = 0.01;
N = 150;
SEED = 1;
OUTPUT = 1;
SKIP = 0;

numSteps = WORDS_LIMIT/WORDS_STEP;

%%%%%%%
%read in the outcomes
%%%%%%%
[WS, DS] = importworddoccounts('docwordcount.txt');
[WO] = textread( 'worddict.txt' , '%s' ); 
inHospDeath = load('inhosp_outcome.txt');
daysToDeath = load('days_outcome.txt');
died = (daysToDeath == 0);
long = (daysToDeath == 1005);
lived = ~(died | long);

% =============================
%     Step2 Topic Modelling
% =============================        
if SKIP == 1
    load 'ldasingle_topics.mat';
else    
    for t = 1:length(NUM_TOPICS)
        topics = NUM_TOPICS(t);
        SEED = SEED + 1;
        ALPHA = 50/topics;

        % Infer the topics and write them to a text file
        eval(['[WP_' num2str(topics) ' ,DP_' num2str(topics) ',Z_' num2str(topics) ' ] = GibbsSamplerLDA( WS , DS , t , N , ALPHA , BETA , SEED , OUTPUT );']);
        eval(['WriteTopics( WP_' num2str(topics) ' , BETA , WO , 10 , 1.0 , 4 , ''topTenWordsPerTopic_' num2str(topics) '.txt'');']);

        % Put the most 10 likely words per topic in cell structure S
        eval(['[S_' num2str(topics) '] = WriteTopics( WP_' num2str(topics) ' , BETA , WO , 10 , 1.0); ' ...
               '[ Order_' num2str(topics) ' ] = OrderTopics( DP_' num2str(topics) ',ALPHA );']);

        %Normalize each row to sum to one
        cmd = ['WDP = DP_' num2str(topics) '; ' ...
               'WDP = bsxfun(@rdivide,WDP,sum(WDP,2)); features_' num2str(topics) ' = WDP; ' ...
               'WDP = WDP''; ' ...
               'WDP = bsxfun(@rdivide,WDP,sum(WDP,2)); ' ...
               'OT_' num2str(topics) ' = WDP * outcomes; '];
        eval(cmd);

        eval(['save(ldasingle_topics.mat, features_' num2str(topics) ', ''-append'')']);
        eval(['dlmwrite(''featurefile_' num2str(topics) '.txt'', full(features_' num2str(topics) '), ''precision'', ''%.6f'');']);
        eval(['save ldasingle_topics' num2str(topics) '.mat WP_' num2str(topics) ' DP_' num2str(topics) ' Z_' num2str(topics) ' OT_' num2str(topics) ' ALPHA BETA SEED N S_' num2str(topics) ';']);   

        %plot images
        eval(['topics = bsxfun(@rdivide,DP_' num2str(topics) ',sum(DP_' num2str(topics) ',2));']);
        topicsOfDead = sum(topics(died, :))./sum(died);
        topicsOfLong = sum(topics(long, :))./sum(long);
        topicsOfLive = sum(topics(lived, :))./sum(lived);

        bar(topicsOfDead, 'r'); hold on; bar(topicsOfLive, 'g'); bar(topicsOfLong, 'b');
        xlabel('Topic Number');
        ylabel('Average Frequency of Occurences Per Patient');
        title('Topic Membership');
        legend('Hospital Mortalities', 'Non-Mortalities', 'Censored Non-Mortalities', 'Location', 'NorthOutside');    
        set(gcf, 'Position', [500 1000 1000 600]);
        I = getframe(gcf);
        imwrite(i.cdata, ['visual2' num2str(topics) '.png']);
        close all;

        eval(['clear WP_' num2str(topics) ' DP_' num2str(topics) ' Z_' num2str(topics) ' OT_' num2str(topics) ' WDP S_' num2str(topics) ' Order_' num2str(topics) ';']);

    end % for topics
end %if SKIP         


%%%%%%%
% Run SVM Model for the different numbers of words used, without the topics
%%%%%%%
for i = 0:numSteps
    numWords = i*WORDS_STEP;
    
    %read in the data for non-EH
    fname = ['svmlight_inputs/svmlight_input' num2str(numWords)];
    input = load(fname);
    features = spconvert(input);
    features(features(:, 3) > 91, 3) = 91;    
    
    [numPatients, numFeatures] = size(features);

    %read in the data with EH    
    fname = ['svmlight_inputs/svmlight_inputEH' num2str(numWords)];
    input = load(fname);
    featuresEH = spconvert(input);
    featuresEH(featuresEH(:, 3) > 91, 3) = 91;
    
    [numPatientsEH, numFeaturesEH] = size(featuresEH);

    % =============================
    %     Step2. Crossvalidation
    % =============================
    indices = crossvalind('Kfold', inHospDeath, 10);       
    cp = classperf(inHospDeath);
    for j = 1:10
        test = (indices == j); 
        train = ~test;
        
        % =============================
        %     Step3. Train/Test SVM
        % =============================
        %if we're on the no words file, run the topics models
        if i == 0
            for k = 1:4
                eval(['featTemp = [features features_' num2str(50) '];']);
                eval(['featTempEH = [featuresEH features_' num2str(50) '];']);
                
                %--- Model with topics only
                svmStruct = svmtrain(featTemp(train, :), inHospDeath(train), 'showplot', true);
                C = svmclassify(svmStruct, featTemp(test, :), 'showplot', true);
                classperf(cp, C, test);

                modelSensTopic(i+1, j) = cp.Sensitivity;
                modelSpecTopic(i+1, j) = cp.Specificity;
                modelPPVTopic(i+1, j) = cp.PositivePredictiveValue;
                modelNPVTopic(i+1, j) = cp.NegativePredictiveValue;
                modelAccTopic(i+1, j) = cp.Sensitivity*cp.Prevalence + cp.Specificity*(1 - cp.Prevalence);
                
                %--- Model with topics and EH       
                svmStruct = svmtrain(featTempEH(train, :), inHospDeath(train), 'showplot', true);
                C = svmclassify(svmStruct, featTempEH(test, :), 'showplot', true);
                classperf(cp, C, test);

                modelSensTopic(i+1, j) = cp.Sensitivity;
                modelSpecTopic(i+1, j) = cp.Specificity;
                modelPPVTopic(i+1, j) = cp.PositivePredictiveValue;
                modelNPVTopic(i+1, j) = cp.NegativePredictiveValue;
                modelAccTopic(i+1, j) = cp.Sensitivity*cp.Prevalence + cp.Specificity*(1 - cp.Prevalence);
            end
        end
        
        %--- Model without EH
        svmStruct = svmtrain(full(features(train, 3:4)), inHospDeath(train), 'showplot', true, ...
                            'Kernel_Function', 'rbf', 'RBF_Sigma', 10000);
        C = svmclassify(svmStruct, full(features(test, :))); %, 'showplot', true);
        classperf(cp, C, test);

        modelSens(i+1, j) = cp.Sensitivity;
        modelSpec(i+1, j) = cp.Specificity;
        modelPPV(i+1, j) = cp.PositivePredictiveValue;
        modelNPV(i+1, j) = cp.NegativePredictiveValue;
        modelAcc(i+1, j) = cp.Sensitivity*cp.Prevalence + cp.Specificity*(1 - cp.Prevalence);

        %--- Model with EH
        svmStruct = svmtrain(full(featuresEH(train, :)), inHospDeath(train), 'showplot', true);
        C = svmclassify(svmStruct, full(featuresEH(test, :)), 'showplot', true);
        classperf(cp, C, test);

        modelSensEH(i+1, j) = cp.Sensitivity;
        modelSpecEH(i+1, j) = cp.Specificity;
        modelPPVEH(i+1, j) = cp.PositivePredictiveValue;
        modelNPVEH(i+1, j) = cp.NegativePredictiveValue;
        modelAccEH(i+1, j) = cp.Sensitivity*cp.Prevalence + cp.Specificity*(1 - cp.Prevalence);
        
        close all;        
    end
end


% =============================
%     Step4. Report SVM Performance
% =============================
save allVars.mat modelSens modelSpec modelPPV modelNPV modelAcc ...
     modelSensEH modelSpecEH modelPPVEH modelNPVEH modelAccEH;
 
 

  