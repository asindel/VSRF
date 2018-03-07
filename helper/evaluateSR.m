function out = evaluateSR(results, imlistHigh, opts)
% evaluateSR function to evaluate the super-resolution results (and 
% interpolated results) compared to ground truth data with PSNR and SSIM
%
% INPUTS 
%  results
%    results(i).imgSR        - super-resolved 3-D images
%    results(i).imgInt       - interpolated 3-D images
%  imlistHigh                - path to high-resolution 3-D images (ground
%                              truth)
%  opts                      - additional options
%    .sf                     - super-resolution factor
%    .interpolMethod         - interpolation method for upscaling, e.g.
%                              'tricubic'
%    .usepf                  - perform evaluation in parallel or
%                              sequentially
% OUTPUT
%  out                       - struct containing PSNR and SSIM results
%   out(i).scoresSR          - PSNR and SSIM for super-resolved 3-D images
%   out(i).scoresInt         - PSNR and SSIM for interpolated 3-D images
%
% @author Aline Sindel
%

out(size(results,2)).scoresSR = [];
out(size(results,2)).scoresInt = [];
scaleDim3 = strcmp(opts.interpolMethod,'tricubic');
sf = opts.sf;
if opts.usepf == 1    
    parfor i=1:size(results,2) %n images
        imgSR = results(i).imgSR;
        imgInt = results(i).imgInt;
        imgRef = loadImage(imlistHigh{i});

        %crop image if necessary (if scaling factor and resolution do not
        %match the images will have a border)
        imgSR = imageModcrop(imgSR,sf,scaleDim3);
        imgInt = imageModcrop(imgInt,sf,scaleDim3);
        imgRef = imageModcrop(imgRef,sf,scaleDim3);

        out(i).scoresSR = evaluateImgQuality(imgSR, imgRef);
        out(i).scoresInt = evaluateImgQuality(imgInt, imgRef);  
    end
else
    for i=1:size(results,2) %n images
        imgSR = results(i).imgSR;
        imgInt = results(i).imgInt;
        imgRef = loadImage(imlistHigh{i});

        %crop image if necessary (if scaling factor and resolution do not
        %match the images will have a border)
        imgSR = imageModcrop(imgSR,sf,scaleDim3);
        imgInt = imageModcrop(imgInt,sf,scaleDim3);
        imgRef = imageModcrop(imgRef,sf,scaleDim3);

        out(i).scoresSR = evaluateImgQuality(imgSR, imgRef);
        out(i).scoresInt = evaluateImgQuality(imgInt, imgRef);  
    end
end
end

function scores = evaluateImgQuality(imgPred, imgRef)
% evaluate the image quality based on PSNR and SSIM for the input image
% pair
% normalize intensity to 0,1
% datatype is Int16 with maximum 32767;
maxInt16 = double(intmax('int16'));
imgRefEval = imgRef / maxInt16;
imgPredEval = imgPred / maxInt16;

N = size(imgPred,3);
psnr = zeros(N,1);
ssim = zeros(N,1);
for i=1:N %compute PSNR and SSIM slice-wise
    psnr(i) = computePSNR(imgRefEval(:,:,i),imgPredEval(:,:,i));
    ssim(i) = computeSSIM(imgRefEval(:,:,i),imgPredEval(:,:,i));
end
%compute PSNR and SSIM for the complete volume
scores.PSNR_3D = computePSNR(imgRefEval,imgPredEval);
scores.SSIM_3D = computeSSIM(imgRefEval,imgPredEval);
scores.PSNR = psnr;
scores.SSIM = ssim;
end

function res = computePSNR(img,ref)
% Peak Signal-To-Noise Ratio (PSNR)
% 
% Given the original (ref) and the super-resolved (or interpolated) image 
% (img), this function returns the PSNR. Code is copied from [1]. 
% 
% References:
% [1] R. Timofte, V. De Smet, L. van Gool. Anchored Neighborhood Regression 
% for Fast Example-Based Super- Resolution. ICCV 2013. 

ref = im2double(ref); % original
img = im2double(img); % distorted
E = ref - img; % error signal
%N = numel(E); % Assume the original signal is at peak (|F|=1)
res = 10*log10( numel(E) / sum(E(:).^2) ); % = 1 / (1/N*sum(E(:).^2)!
end

function res = computeSSIM(img,ref)
% Structural Similarity Index (SSIM)
% MATLAB implementation of SSIM 
ref = im2double(ref); % original
img = im2double(img); % super-resolved (or interpolated) image = 'distorted'
res = ssim(img,ref) ;
end