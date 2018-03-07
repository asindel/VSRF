function out = applyVSRF( imlistHigh, imlistLow, srforest)
% applyVSRF Applies the trained volumetric super-resolution forest 
% with 3-D patches to MRI data.
%
% This function applies the trained volumetric super-resolution forest 
% (VSRF) to a set of MR volumes. 
% Either imlistHigh or imlistLow has to be defined (MR volumes as .mat 
% files). If only imlistHigh is defined, downsampling is applied to the MR 
% volumes to generate the low-resolution MR volumes.
%
% USAGE
%  opts = srForestApply( )
%  out  = srForestApply( imlistHigh, imlistLow, srforest )
%
% INPUTS
%  imlistHigh   - data path to high-resolution volumes (.mat).
%  imlistLow    - data path to low-resolution volumes (.mat). If imlistLow  
%                  is empty and imlistHigh is provided, the HR volumes are 
%                  downscaled.
%  srforest     - trained volumetric super-resolution forest 
%                 (see srForestTrain3D.m)
%
% OUTPUTS
%  out          - struct array with output information for each image
%   .imgSR       - super-resolved volume
%   .imgInt      - interpolated volume
%
% See also: trainVSRF
%
% The code is adapted from [1] and Piotr's Image&Video Toolbox (Copyright Piotr
% Dollar).
%
% @author Aline Sindel
%
% REFERENCES
% [1] S. Schulter, C. Leistner, and H. Bischof. Fast and accurate image 
% upscaling with super-resolution forests. CVPR 2015.

if ~isfield(srforest.sropts, 'Mhat'),srforest.sropts.Mhat = []; end
if ~isfield(srforest.sropts, 'nthreads'),srforest.sropts.nthreads = 1; end
if isempty(srforest.sropts.ensemble), srforest.sropts.ensemble='median'; end

% check input
if isempty(imlistLow)&&isempty(imlistHigh), error('Either dataLow or dataHigh has to be provided'); end
if isempty(srforest), error('the model srForest has to be provided'); end
if isempty(srforest.sropts.Mhat), srforest.sropts.Mhat=length(srforest.model); end

% check if we need to downscale the high-res images first for evaluation!
downscale = false; if isempty(imlistLow), downscale=true; imlistLow = imlistHigh; end

if ~isempty(imlistLow)    
    nImgs=length(imlistLow);
else
    nImgs=0; 
end

out(nImgs).imgSR = [];
out(nImgs).imgInt = [];
% iterate the low-res (or to be downscaled high-res) 3D images and upscale them
for i=1:nImgs
    if srforest.sropts.verbose, fprintf('Upscale image %d/%d\n',i,nImgs); end
    % load low-res (or to be downscaled high-res) 3D images
    [imgL, voxelspacing] = loadImage(imlistLow{i}); 

    % preprocessing
    imgL = im2single(imgL);     
    
    if downscale
        sizeImgOrig = size(imgL);
        imgL = imageModcrop(imgL, srforest.sropts.sf, srforest.sropts.scaleDim == 3);
        sizeImgCropped = size(imgL); 
        imgL = imageDownsampling(imgL, 1/srforest.sropts.sf, srforest.sropts.downsamplingMethod);
    else
        if srforest.sropts.scaleDim == 3
            voxelspacing = voxelspacing/srforest.sropts.sf;
        else
            voxelspacing = voxelspacing./[srforest.sropts.sf, srforest.sropts.sf, 1]';
        end
    end
    
    % upsampling to generate the mid-res image
    imgL = imageUpsampling(imgL, srforest.sropts.sf, srforest.sropts.interpolMethod);
    
    % generate volumetric super-resolution forest output
    imgSR = vsrfApply(imgL,srforest,srforest.sropts.Mhat,srforest.sropts.nthreads, voxelspacing);    

    imgSR = im2double(imgSR);
    imgL = im2double(imgL);    
    if downscale && any(sizeImgOrig ~= sizeImgCropped)
        imgSR_out = zeros(sizeImgOrig,'double');
        imgSR_out(1:sizeImgCropped(1), 1:sizeImgCropped(2),1:sizeImgCropped(3)) = imgSR;
        imgInt_out = zeros(sizeImgOrig,'double');
        imgInt_out(1:sizeImgCropped(1), 1:sizeImgCropped(2),1:sizeImgCropped(3)) = imgL;

        out(i).imgSR = imgSR_out;
        out(i).imgInt = imgInt_out;
    else
		out(i).imgSR = imgSR;
        out(i).imgInt = imgL;
    end
end
end

function imout = vsrfApply( imM, srforest, Mhat, nthreads, voxelspacing )
% set some constants
opts = srforest.sropts;
tarBorder = [0 0 0];
% extract patches and compute features
patchfeats = opts.patchfeats; %local copy
filters = patchfeats.filters;
if numel(filters)>4
    %compute z-weight for 3D filtering
    zWeight = voxelspacing(1)/voxelspacing(3); %assumption: voxelspacing(1)==voxelspacing(2)
    for i=3:3:numel(filters)
        filters{i} = filters{i}*zWeight;
    end
    patchfeats.filters = filters;
end
patchesSrc = extractPatches3D(imM,opts.patchSize,...
    opts.patchStride,tarBorder,patchfeats);
patchesSrc = srforest.Vpca' * patchesSrc;

% apply random regression forest
patchesTarPred = forestRegrApply(patchesSrc,patchesSrc,...
    srforest.model,srforest.sropts.pRegrForest.leaflearntype,Mhat,nthreads);
patchesTarPred = cat(3,patchesTarPred{:});
%ensemble model:
if strcmp(opts.ensemble, 'average')
     patchesTarPred = sum(patchesTarPred,3)/size(patchesTarPred,3);
elseif strcmp(opts.ensemble, 'median')
    patchesTarPred = median(patchesTarPred,3); 
else
    error('Unknown ensemble model type!');
end

% add mid-res patches + patches predicted by SRF
patchesMid = extractPatches3D(imM,opts.patchSize,...
    opts.patchStride,tarBorder);
patchesTarPred = patchesTarPred + patchesMid;

% merge patches into the final output (i.e., average overlapping patches)
img_size = size(imM); 
grid = getVectorizedSamplingGrid(img_size,...
    opts.patchSize,opts.patchStride,tarBorder);
imout = overlap_add(patchesTarPred,img_size,grid);

%compute cut off part (if patch size does not fit to image size)
patch_size = opts.patchSize;
overlapstep = opts.patchStride;
for k=1:3
    nPatch = floor((img_size(k)-patch_size(k))/overlapstep(k));
    posEnd = nPatch*overlapstep(k) + patch_size(k) + 1;
    % add cut off part from mid-res image
    if nPatch<=img_size(k)
        if k==1
            imout(posEnd:img_size(k),:,:) = imM(posEnd:img_size(k),:,:);
        elseif k==2
            imout(:,posEnd:img_size(k),:) = imM(:,posEnd:img_size(k),:);
        elseif k==3
            imout(:,:,posEnd:img_size(k)) = imM(:,:,posEnd:img_size(k));
        end
    end
end

end

function result = overlap_add( patches, img_size, grid )
% Image construction from overlapping patches
result = zeros(img_size,'single');
weight = zeros(img_size);
for i = 1:size(grid,2)
    result(grid(:,i)) = result(grid(:,i)) + patches(:,i);
    weight(grid(:,i)) = weight(grid(:,i)) + 1;
end
I = logical(weight);
result(I) = result(I) ./ weight(I);
end
