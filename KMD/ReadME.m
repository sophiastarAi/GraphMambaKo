% KMD of PM2.5

%% Instructions
% To generate Koopman modes call on the function:
% Modes=GenerateKoopmanModes(Data,Mode1,Mode2,Save)

%Inputs Required:
% Call string flags of different datasets

% Mode1 and Mode2 are integers indicating which modes to produce.

% Below is the description of the dataset:
% 1. 2018-2019-hourly PM2.5 per hour from 2018 to 2019
% 2. 2019-hourly PM2.5 per hour in 2019
% 3. 121 Dimensions of monitoring stations

%Outputs:
% Modes is an n by m by #modes sized  array. 
% For example Modes(:,:,i) contains the i'th mode.

%Plots Generated:
% The funciton will generate plots of the desired Koopman Modes.

% Examples:1
clc; clear variables; close all;

Daily=GenerateKoopmanModes('Day_mean',1,20)





