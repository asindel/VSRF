function patches = extractPatches3D( img, patchsize, overlapstep, border, opts )
% extractPatches3D Extracts 3-D patches from one MR volume
% 
% The function extract all possible 3-D patches from one MR volume 
% considering the setting for patch extraction (patchsize, overlapstep, 
% border and features). Compared to [1], patches and features
% are extended to 3-D and are extracted from a 3-D MR volumes and an 
% additional feature set filtersDevEdge is provided.
% 
% INPUTS
%  img          - [imgh x imgw x imgd] 3-D image (MR volume)
%  patchsize    - [ph x pw x pd] size of the patches to be extracted
%  overlapstep  - [oy x ox x od] overlapstep of neighboring patches. 
%                 The minimum selectable overlapstep is [2 x 2 x 2] and the 
%                 maximum overlapstep is [ph x pw x pd] - [1 x 1 x 1]. 
%  border       - [by x bx x bz] border of images that is left out
%  opts         - additional params (struct)
%   .type       - ['none'] what kind of features to compute. Alternative
%                 options are 'filters' and 'filtersDevEdge', where also 
%                 the .filters options should be set. 
%   .filters    - [] filter kernels (for .type='filters'). This should be a
%                 a cell array with different filters. 
% 
% OUTPUTS
%  patches      - [ph*pw*pd*nfilters x npatches] extracted patches
%                 (vectorized). 
%                 For filters: nfilters = numel(.filters), 
%                 for filtersDevEdge: nfilters = numel(.filters)+4; 
%                 npatches defines the number of patches that could be 
%                 extracted with the given settings. 
% 
% See also getVectorizedSamplingGrid
% 
% Code adapted from [1] and [2]
% 
% @author Aline Sindel
%
% References:
% [1] S. Schulter, C. Leistner, and H. Bischof. Fast and accurate image 
% upscaling with super-resolution forests. CVPR 2015.
% [2] R. Timofte, V. De Smet, L. van Gool. Anchored Neighborhood Regression 
% for Fast Example-Based Super- Resolution. ICCV 2013. 

if nargin < 5, opts = struct; end
if ~isfield(opts, 'type'),opts.type = 'none'; end
if ~isfield(opts, 'filters'),opts.filters = []; end
if nargin == 0, patches=opts; return; end

% check input
if any(patchsize<3), error('patchsize shoud be >= 3'); end

% Compute one grid for all filters
grid = getVectorizedSamplingGrid(size(img),patchsize,overlapstep,border);
switch opts.type
    case 'none'
        patches = img(grid); 
    case 'filters' %gradient + 2nd dev
        prodPatchsize = prod(patchsize);
        feature_size = prodPatchsize*(numel(opts.filters));
        patches = zeros([feature_size,size(grid,2)],'single');
        for i = 1:numel(opts.filters)
            fImg = convn(img,opts.filters{i},'same');            
            patches((1:prodPatchsize)+(i-1)*prodPatchsize,:) = fImg(grid);
        end  
    case 'filtersDevEdge' %1st dev + 2nd dev + edge orientation + magnitude
        prodPatchsize = prod(patchsize);
        feature_size = prodPatchsize*(numel(opts.filters)+4);
        patches = zeros([feature_size,size(grid,2)],'single');
        
        %1st and 2nd derivative
        for i = 1:numel(opts.filters)
            fImg = convn(img,opts.filters{i},'same');            
            patches((1:prodPatchsize)+(i-1)*prodPatchsize,:) = fImg(grid);
        end
        imgG = imgaussfilt3(img,1);
        %edge orientation
        gImgX = convn(imgG,opts.filters{1},'same');
        gImgY = convn(imgG,opts.filters{2},'same');
        gImgZ = convn(imgG,opts.filters{3},'same'); 
        for i = 1:3
            switch i
                case 1 %Gy/Gx
                    fImg = atan(gImgY./gImgX); 
                case 2 %Gx/Gz
                    fImg = atan(gImgX./gImgZ);
                case 3 %Gy/Gz
                    fImg = atan(gImgY./gImgZ);                     
            end
            %set nan to zero, Gy/Gx = nan if Gy=Gx=0
            nanI = isnan(fImg);
            A = zeros(size(fImg));
            if sum(nanI(:))>0
                idx = logical(nanI);
                fImg(idx) = A(idx);
            end  
            patches((1:prodPatchsize)+(i+numel(opts.filters)-1)*prodPatchsize,:) = fImg(grid);                           
        end
        
        i=i+numel(opts.filters)+1;
        %edge magnitude
        fImg = sqrt(gImgX.^2 + gImgY.^2 + gImgZ.^2);        
        patches((1:prodPatchsize)+(i-1)*prodPatchsize,:) = fImg(grid);
    otherwise
        error('Unknown features');
end
end


