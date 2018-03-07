function main_VSRF_MRI(sropts,dataopts)
% Function to perform MRI Super-Resolution with volumetric super-resolution
% forests (VSRF)
% If dataopts.train = false a trained forest model is loaded and the forest 
% is only applied.
%
% INPUTS
% sropts   - struct with additional super-resolution options, all options 
%            that are not set by the user are set to the default parameters
% dataopts - additional options for the data to select and if the forest
%            should be trained or a trained model should be loaded instead
%
% @author Aline Sindel
%
if ~isfield(dataopts, 'train'),dataopts.train = true; end %default train forest
if ~isfield(dataopts, 'testHR'),dataopts.testHR = true; end %required for evaluation
if ~isfield(dataopts, 'testLR'),dataopts.testLR = false; end %optional, if false then downsampling
if ~isfield(dataopts, 'trainLR '),dataopts.trainLR  = false; end %optional, if false then downsampling

addpath('helper/'); addpath('method/'); addpath('helper/MatlabFileExchange/');

%%%%%%%%%%%%%%% adapt path to SPM %%%%%%%%%%%%%%%%%%%%
% pathSPM = 'pathToSPM';
% if exist(pathSPM, 'dir')
%     addpath(pathSPM); %if path is not added to Matlab path
% end

%%%%%%%%%%%%%% adapt data path - start folder for MR data selection
datapath = pwd;

%paths
srforestmPath = 'method/models';
outdir = strcat(pwd,'\results\',datestr(now,'yyyy_mm_dd_HH_MM_SS'));
mkdir(outdir);

%%%% PARAMETERS %%%%
train = dataopts.train; 
testHR = dataopts.testHR; 
testLR = dataopts.testLR; 
trainLR = dataopts.trainLR; 

SRmethod = 'VSRF';

% set remaining options to default
opts = setParams(sropts); 


opts.usepf = opts.pRegrForest.usepf==1;
if opts.usepf==1
    opts.nthreads = feature('numcores')*2; %inference parallel
else
    opts.nthreads = 1;
end

%%%% SELECT INPUT FILES %%%%%
    imTestLow = {}; imTestHigh = {};
    if testLR == true
        inputTestLRFolders = uipickfiles('Prompt', 'Select input LR TEST folders', ...
        'FilterSpec', datapath);
        imTestLow = getInputFiles(inputTestLRFolders);
    end
    if testHR == true       
        inputTestHRFolders = uipickfiles('Prompt', 'Select input HR TEST folders', ...
        'FilterSpec', datapath);    
        imTestHigh = getInputFiles(inputTestHRFolders);
    end    


if train == true
    imTrainLow = {}; 
    if trainLR == true
        inputTrainLRFolders = uipickfiles('Prompt', 'Select input LR TRAIN folders', ...
        'FilterSpec', datapath);
        imTrainLow = getInputFiles(inputTrainLRFolders);
    end
    inputTrainHRFolders = uipickfiles('Prompt', 'Select input HR TRAIN folders', ...
    'FilterSpec', datapath);    
    imTrainHigh = getInputFiles(inputTrainHRFolders);            
    % create path to the model file
    srforestFNm = sprintf('%s_%dD_sf-%d_T-%02d.mat',SRmethod,opts.scaleDim, opts.sf,...
      opts.pRegrForest.M);
    srforestFNm = fullfile(srforestmPath,srforestFNm);
else
    % select path to the model file
    [srforestFNm, srforestFNmPath]= uigetfile(srforestmPath,sprintf('Select %s-model',SRmethod));
    srforestFNm = fullfile(srforestFNmPath,srforestFNm);
end


% output textfile name 
method = regexprep(opts.interpolMethod,'(\<\w)','${upper($1)}');
textfilename = fullfile(outdir,sprintf('output_%s_%dD_sf-%d_T-%02d.txt',SRmethod,opts.scaleDim,opts.sf,opts.pRegrForest.M));
tablefilename = fullfile(outdir,sprintf('table_%s_%dD_sf-%d_T-%02d.csv',SRmethod,opts.scaleDim,opts.sf,opts.pRegrForest.M));
tablefilenameInterpol = fullfile(outdir,sprintf('table_%sInterpol_sf-%d.csv',method,opts.sf));

fileID = fopen(textfilename,'w');
mfprintf(fileID, 'Volumetric Super-Resolution Forest\n');
if train == true
    mfprintf(fileID, 'LR-Train images: ');
    for i=1:size(imTrainLow,2)
    mfprintf(fileID, '%s, ', imTrainLow{i});
    end
    mfprintf(fileID, '\nHR-Train images: ');
    for i=1:size(imTrainHigh,2)
    mfprintf(fileID, '%s, ', imTrainHigh{i});
    end
    mfprintf(fileID, '\n');
end
mfprintf(fileID, 'LR-Test images: ');
for i=1:size(imTestLow,2)
mfprintf(fileID, '%s, ', imTestLow{i});
end
mfprintf(fileID, '\nHR-Test images: ');
for i=1:size(imTestHigh,2)
mfprintf(fileID, '%s, ', imTestHigh{i});
end
mfprintf(fileID, '\n');
mfprintf(fileID, '%dD-SR\nsf:%d\nInterpol:%s\nDownsampling:%s\nT%d\nFilters:%s\nEnsembleModel:%s\npf:%d\nNthreads:%d\n', ...
    opts.scaleDim,opts.sf, opts.interpolMethod, opts.downsamplingMethod, opts.pRegrForest.M, opts.patchfeats.type, opts.ensemble, opts.usepf,opts.nthreads);
tStart = tic;
%%%% Start Training %%%%
if train == true
    nTrainImgs = size(imTrainHigh,2);
    if size(imTrainLow,2)>nTrainImgs, error('Number of LR and HR images must be the same (or only HR images)!'); end 
    mfprintf(fileID, 'Training volumetric super-resolution forest #imgs=%d\n', nTrainImgs);  
    srforest = trainVSRF(opts, imTrainHigh, imTrainLow);
    tTrain = toc(tStart);
    mfprintf(fileID, 'train-time: %2.4f s\n', tTrain);
    saveVSRF(srforestFNm,srforest);
else
    mfprintf(fileID, 'Loading volumetric super-resolution forest: %s\n', srforestFNm);
    srforest = loadVSRF(srforestFNm);
end

srforest.sropts.ensemble = opts.ensemble;

%%%% Start Testing %%%%
if testHR==true&&testLR==true&&size(imTestHigh,2)~=size(imTestLow,2)
    error('Number of LR and HR images must be the same (or only HR images)!'); 
end 
nTestImgs = max(size(imTestHigh,2),size(imTestLow,2));
mfprintf(fileID, 'Testing volumetric super-resolution forest #imgs=%d\n', nTestImgs);
tTestStart = tic;  
results = applyVSRF(imTestHigh, imTestLow,srforest);
tTest = toc(tTestStart);
mfprintf(fileID, 'test-time: %2.4f s\n', tTest);
tEvalStart = tic;

%%%%%%% parallel evaluation %%%%%%% set opts.usepf = 1 %%%%%%%%%%%%%%%%
opts.usepf = 1;

%%%% Evaluate the super-resolution forest %%%%

fileIDTable = fopen(tablefilename,'w');
fileIDTableInterpol = fopen(tablefilenameInterpol,'w');
if testHR==true
    stats = evaluateSR(results, imTestHigh, opts);  
    tEval = toc(tEvalStart);
    mfprintf(fileID, 'evaluation-time: %2.4f s\n', tEval);

    tTotal = toc(tStart);
    mfprintf(fileID, 'total time: %2.4f s\n', tTotal);

    % visualize some statistics
    mfprintf(fileID, 'Test results - Volumetric super-resolution forest\n');
    meanPSNR_SR = 0;
    meanSSIM_SR = 0;
    meanPSNR_Int = 0;
    meanSSIM_Int = 0;
    
    for i=1:nTestImgs
        evalSR = stats(i).scoresSR;
        evalInt = stats(i).scoresInt;
        if ~isempty(imTestLow)
            mfprintf(fileID, 'FILE: %s\n', imTestLow{i});
            fprintf(fileIDTable, 'FILE: %s\n', imTestLow{i});
            fprintf(fileIDTableInterpol, 'FILE: %s\n', imTestLow{i});
        else
            mfprintf(fileID, 'FILE: %s\n', imTestHigh{i});
            fprintf(fileIDTable, 'FILE: %s\n', imTestHigh{i});
            fprintf(fileIDTableInterpol, 'FILE: %s\n', imTestHigh{i});
        end
        mfprintf(fileID, '%s Upsampling:    3D-Img %d/%d PSNR = %.2f dB, SSIM = %.4f \n', SRmethod,i, nTestImgs, evalSR.PSNR_3D, evalSR.SSIM_3D);
        mfprintf(fileID, '%s Upsampling: 3D-Img %d/%d PSNR = %.2f dB, SSIM = %.4f \n', method, i, nTestImgs, evalInt.PSNR_3D, evalInt.SSIM_3D);
        
        fprintf(fileIDTable, 'MeanPSNR; MeanSSIM\n%.2f; %.4f\n', evalSR.PSNR_3D, evalSR.SSIM_3D);
        fprintf(fileIDTableInterpol, 'MeanPSNR; MeanSSIM\n%.2f; %.4f\n', evalInt.PSNR_3D, evalInt.SSIM_3D);
        
        meanPSNR_SR = meanPSNR_SR + evalSR.PSNR_3D;
        meanSSIM_SR = meanSSIM_SR + evalSR.SSIM_3D;
        meanPSNR_Int = meanPSNR_Int + evalInt.PSNR_3D;
        meanSSIM_Int = meanSSIM_Int + evalInt.SSIM_3D;

        psnr_SR = evalSR.PSNR;
        psnr_Int = evalInt.PSNR;
        ssim_SR = evalSR.SSIM;
        ssim_Int = evalInt.SSIM;

        fprintf(fileIDTable, 'psnr; ssim\n');
        fprintf(fileIDTableInterpol, 'psnr; ssim\n');
        nSlices = size(psnr_SR,1);
        for z=1:nSlices
            mfprintf(fileID, '%s Upsampling:    3D-Img %d/%d (slice %d/%d): psnr = %.2f dB, ssim = %.4f \n',SRmethod,i,nTestImgs,z,nSlices,psnr_SR(z),ssim_SR(z));
            mfprintf(fileID, '%s Upsampling: 3D-Img %d/%d (slice %d/%d): psnr = %.2f dB, ssim = %.4f \n', method,i,nTestImgs,z,nSlices,psnr_Int(z),ssim_Int(z));
            
            fprintf(fileIDTable, '%.2f; %.4f\n',psnr_SR(z),ssim_SR(z));
            fprintf(fileIDTableInterpol, '%.2f; %.4f\n',psnr_Int(z),ssim_Int(z));  
        end
    end
    mfprintf(fileID, '%s Upsampling:    Mean-PSNR = %.2f dB, Mean-SSIM = %.4f \n', SRmethod,meanPSNR_SR/nTestImgs, meanSSIM_SR/nTestImgs);
    mfprintf(fileID, '%s Upsampling: Mean-PSNR = %.2f dB, Mean-SSIM = %.4f \n', method,meanPSNR_Int/nTestImgs, meanSSIM_Int/nTestImgs);
    
    fprintf(fileIDTable, 'MeanMeanPSNR; MeanMeanSSIM\n%.2f; %.4f\n', meanPSNR_SR/nTestImgs, meanSSIM_SR/nTestImgs);
    fprintf(fileIDTableInterpol, 'MeanMeanPSNR; MeanMeanSSIM\n%.2f; %.4f\n', meanPSNR_Int/nTestImgs, meanSSIM_Int/nTestImgs);
end

fclose(fileID);
fclose(fileIDTable);
fclose(fileIDTableInterpol);

%%%% Save result images and visualize them (optional) %%%%
for i=1:nTestImgs
    %%% For visualization uncomment the lines
    imgSR = results(i).imgSR;
    imgInt = results(i).imgInt;    
%     imgDiffSRInt = imabsdiff(imgSR,imgInt); 
%     if ~isempty(imTestHigh)  
%         imgGT = loadImage(imTestHigh{i});        
%         imgDiffGTInt = imabsdiff(imgGT,imgInt);
%         imgDiffGTSR = imabsdiff(imgGT,imgSR);
%     end

    %%% Save images
    if ~isempty(imTestLow)        
        fileOrig = strcat(imTestLow{i});
    else        
        fileOrig = strcat(imTestHigh{i});
    end
    saveImage(imgSR, fileOrig, outdir, opts.sf, SRmethod);
    saveImage(imgInt, fileOrig, outdir, opts.sf, method);
%     
%     close all;
%     h1 = figure; imshow3Dfull(imgInt, [0 32767]); set(h1, 'name', sprintf('%s Interpolation', method)); 
%     h2 = figure; imshow3Dfull(imgSR, [0 32767]); set(h2, 'name', sprintf('%s Reconstruction', SRmethod));
%     h3 = figure; imshow3Dfull(imgDiffSRInt, [0 32767]); set(h3, 'name', 'Difference SR - Interpolation');  
%     
%     if ~isempty(imTestHigh)  
%         h4 = figure; imshow3Dfull(imgGT,[0 32767]); set(h4, 'name', 'Ground Truth');  
%         h5 = figure; imshow3Dfull(imgDiffGTInt, [0 32767]); set(h5, 'name', 'Difference GT - Interpolation');        
%         h6 = figure; imshow3Dfull(imgDiffGTSR, [0 32767]); set(h6, 'name', 'Difference GT - SR');  
%     end

end

