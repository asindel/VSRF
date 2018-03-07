%%% Volumetric Super-Resolution Forest (VSRF)
% Example script to train and apply VSRF
% optional parameters are set in comments (see setParams.m for a detailed
% information of the default parameters)
%
% @author Aline Sindel
%

addpath('helper/'); addpath('method/'); 


% parameters are set by default (see setParams.m), adapt here only SR-options which should be
% different from the default parameters

sropts.sf = 2; %super-resolution factor


% tree parameters
sropts.pRegrForest = forestRegrTrain(); %previous to adaptation of tree parameters, always run this line (default)
sropts.pRegrForest.M = 30; %number of trees, 30 is default
sropts.pRegrForest.usepf = 1; %MATLAB parallel workers, matlabpool open required! 
% sropts.pRegrForest.Npworkers = 2; %specify number of matlab parallel workers

%%%% PARAMETERS %%%%
dataopts.train = true; %set to false if no training required!
% dataopts.testHR = true; %required for evaluation
% dataopts.testLR = false; %optional, if false then downsampling
% dataopts.trainLR = false; %optional, if false then downsampling

%%% Call main_VSRF_MRI with parameters sropts, dataopts
main_VSRF_MRI(sropts,dataopts);

%%% user: Specify paths in main_VSRF_MRI.m
