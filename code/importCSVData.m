function [labels, features, icuStay, hospStay, survivalTime, hospExpireFlag, readmit, totalHospStays, sid] = importCSVData(fname)

% =============================
%     Step1. Data Scanning
% =============================

%
% path to read the input csv file
fid=fopen(fname);

SUBJECT_ID = 1;
SEX = 2;
DOB = 3;	
DOD = 4;
HOSPITAL_EXPIRE_FLG = 5;
DAYSFROMFIRSTDISCHARGETODEATH = 6;	
ICUSTAY_LOS	= 7;
HOSPITAL_LOS = 8;
ICUSTAY_ADMIT_AGE = 9;	
SAPSI_FIRST	= 10;
SOFA_FIRST = 11;
SAPSI_MIN = 12;
SAPSI_MAX = 13;
SAPSI_FINAL = 14;
ELIX_INDEX = 14;
HOSPITAL_TOTAL_NUM = ELIX_INDEX + 30 + 1;
HOSPITAL_ADMIT_DATE = ELIX_INDEX + 30 + 2;
ICUSTAY_INTIME = ELIX_INDEX + 30 + 3;
FIRSTDISCHARGEDATE = ELIX_INDEX + 30 + 4;
SECONDADMITDATE = ELIX_INDEX + 30 + 5;
DAYSFROMFIRSTDISCHTOREADMIT = ELIX_INDEX + 30 + 6;


% 45 columns
scanStr = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s';                     
text=textscan(fid, scanStr, 'delimiter', '\t');
fclose(fid);

% ==============================
%     Step2. Data Retrieval
% ==============================

%demographics
sid = cellfun(@str2num, text{SUBJECT_ID});
gender = zeros(length(sid), 1); gender(strcmp(text{SEX}, '"M"')) = 1; 
age = cellfun(@str2num, text{ICUSTAY_ADMIT_AGE});

data = cellfun(@str2num, text{SAPSI_FIRST}, 'UniformOutput', false); 
data(any(cellfun('isempty',data),2),:) = {NaN}; 
saps = cell2mat(data);

data = cellfun(@str2num, text{SAPSI_MIN}, 'UniformOutput', false); 
data(any(cellfun('isempty',data),2),:) = {NaN}; 
sapsMin = cell2mat(data);

data = cellfun(@str2num, text{SAPSI_MAX}, 'UniformOutput', false); 
data(any(cellfun('isempty',data),2),:) = {NaN}; 
sapsMax = cell2mat(data);

data = cellfun(@str2num, text{SAPSI_FINAL}, 'UniformOutput', false); 
data(any(cellfun('isempty',data),2),:) = {NaN}; 
sapsFinal = cell2mat(data);
 
data = cellfun(@str2num, text{SOFA_FIRST}, 'UniformOutput', false); 
data(any(cellfun('isempty',data),2),:) = {NaN}; 
sofa = cell2mat(data);

icuStay = cellfun(@str2num,text{ICUSTAY_LOS	});
hospStay = cellfun(@str2num,text{HOSPITAL_LOS}); 

data = cellfun(@str2num, text{DAYSFROMFIRSTDISCHARGETODEATH}, 'UniformOutput', false); 
data(any(cellfun('isempty',data),2),:) = {1005}; %--> Set the survival time to 1005
survivalTime = cell2mat(data);

hospExpireFlag = logical(cellfun(@str2num, text{HOSPITAL_EXPIRE_FLG}));

data = cellfun(@str2num, text{DAYSFROMFIRSTDISCHTOREADMIT}, 'UniformOutput', false); 
data(any(cellfun('isempty',data),2),:) = {NaN};
readmit = cell2mat(data);

totalHospStays = cellfun(@str2num, text{HOSPITAL_TOTAL_NUM});


%Elix-Hauser Scores
for i = 1:30
    cmd = ['E' num2str(i) ' = cellfun(@str2num, text{' num2str(i+ELIX_INDEX) '});'];
    eval(cmd); %E1 -> 15, E30 -> 44
end

%adjust old ago to median old age
age(age >= 200) = 91;

%only keep the positives that have SSRI or NSRI meds
features = [gender age saps sapsMin sapsMax sapsFinal sofa ...                         
             E1 E2 E3 E4 E5 E7 E8 E9 E10 ...
             E11 E12 E13 E14 E15 E16 E17 E18 E19 E20 E21 E22 E23 E24 E25 ...
             E26 E27 E28 E29 E30];

labels = {'Gender', 'Age', 'SAPS', 'SAPS_Min', 'SAPS_Max', 'SAPS_Final', ...
              'SOFA', 'Congestive_Heart_Failure' 'Cardiac_Arrhythmias' ...
              'Valvular_Disease' 'AIDS' 'Alcohol_Abuse' 'Chronic_Pulmonary' ...
              'Coagulopathy' 'Deficiency_Anemias' 'Depression' ...
              'Diabetes_Complicated' 'Diabetes_Uncomplicated' 'Drug_Abuse' ...
              'Fluid_Electrolyte' 'Hypertension' 'Hypothyroidism' ...
              'Liver_Disease' 'Lymphoma' 'Metastatic_Cancer' 'Obesity' ...
              'Other_Neurological' 'Paralysis' 'Peptic_Ulcer' ...
              'Peripheral_Vascular' 'Psychoses' 'Pulmonary_Circulation' ...
              'Renal_Failure' 'Rheumatoid_Arthritis' 'Solid_Tumor' 'Weight_Loss'};  
     