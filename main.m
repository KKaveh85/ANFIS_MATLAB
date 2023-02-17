%
% predictions of suspended sediment 

clc;
clear;
close all;

%% Create Time-Series Data

data = load('data');

% Input Dataset
% Inputs:
%     1. Flows
% 
% Targets:
%     1. Suspended sediment concentration
%    

Inputs = data.Inputs';
Targets = data.Targets';

nData = size(Inputs,1);

Targets = Targets(:,1); % Select 1st Output to Model

%% Shuffling Data

PERM = randperm(nData); % Permutation to Shuffle Data

pTrain=0.85;
nTrainData=round(pTrain*nData);
TrainInd=PERM(1:nTrainData);
TrainInputs=Inputs(TrainInd,:);
TrainTargets=Targets(TrainInd,:);

pTest=1-pTrain;
nTestData=nData-nTrainData;
TestInd=PERM(nTrainData+1:end);
TestInputs=Inputs(TestInd,:);
TestTargets=Targets(TestInd,:);

%% Selection of FIS Generation Method

 Option{1}='Grid Partitioning (genfis1)';
 Option{2}='Subtractive Clustering (genfis2)';
Option{3}='FCM (genfis3)';

ANSWER=questdlg('Select FIS Generation Approach:',...
                'Select GENFIS',...
                  Option{1},Option{2},Option{3},...
                Option{3});
pause(0.01);

%% Setting the Parameters of FIS Generation Methods

switch ANSWER
     case Option{1}
         Prompt={'Number of MFs','Input MF Type:','Output MF Type:'};
         Title='Enter genfis1 parameters';
         DefaultValues={'5', 'gaussmf', 'linear'};
         
         PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
         pause(0.01);
 
         nMFs=str2num(PARAMS{1});	%#ok
         InputMF=PARAMS{2};
         OutputMF=PARAMS{3};
         
         fis=genfis1([TrainInputs TrainTargets],nMFs,InputMF,OutputMF);
 
     case Option{2}
         Prompt={'Influence Radius:'};
         Title='Enter genfis2 parameters';
         DefaultValues={'0.3'};
         
         PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
         pause(0.01);
 
         Radius=str2num(PARAMS{1});	%#ok
        
         fis=genfis2(TrainInputs,TrainTargets,Radius);
         
    case Option{3}
        Prompt={'Number fo Clusters:',...
                'Partition Matrix Exponent:',...
                'Maximum Number of Iterations:',...
                'Minimum Improvemnet:'};
        Title='Enter genfis3 parameters';
        DefaultValues={'15', '2', '200', '1e-5'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.01);

        nCluster=str2num(PARAMS{1});        %#ok
        Exponent=str2num(PARAMS{2});        %#ok
        MaxIt=str2num(PARAMS{3});           %#ok
        MinImprovment=str2num(PARAMS{4});	%#ok
        DisplayInfo=1;
        FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];
        
        fis=genfis3(TrainInputs,TrainTargets,'sugeno',nCluster,FCMOptions);
end

%% Training ANFIS Structure

Prompt={'Maximum Number of Epochs:',...
        'Error Goal:',...
        'Initial Step Size:',...
        'Step Size Decrease Rate:',...
        'Step Size Increase Rate:'};
Title='Enter genfis3 parameters';
DefaultValues={'200', '0', '0.01', '0.9', '1.1'};

PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
pause(0.01);

MaxEpoch=str2num(PARAMS{1});                %#ok
ErrorGoal=str2num(PARAMS{2});               %#ok
InitialStepSize=str2num(PARAMS{3});         %#ok
StepSizeDecreaseRate=str2num(PARAMS{4});    %#ok
StepSizeIncreaseRate=str2num(PARAMS{5});    %#ok
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid

fisout=anfis([TrainInputs TrainTargets],fis,TrainOptions,DisplayOptions,[],OptimizationMethod);


%% Apply ANFIS to Data

Outputs=evalfis(Inputs,fisout);
TrainOutputs=Outputs(TrainInd,:);
TestOutputs=Outputs(TestInd,:);

%% Error Calculation

TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors.^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors.^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

%% Plot Results

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

figure;
PlotResults(TestTargets,TestOutputs,'Test Data');

figure;
PlotResults(Targets,Outputs,'All Data');

if ~isempty(which('plotregression'))
    figure;
    plotregression(TrainTargets, TrainOutputs, 'Train Data', ...
                   TestTargets, TestOutputs, 'Test Data', ...
                   Targets, Outputs, 'All Data');
    set(gcf,'Toolbar','figure');
end

% figure;
% gensurf(fis, [1], 1, [30]);
% xlim([min(Inputs(:,1)) max(Inputs(:,1))]);
% % ylim([min(Inputs(:,2)) max(Inputs(:,2))]);
        
