function grid = getVectorizedSamplingGrid(img_size, patch_size, overlapstep, border)
% getVectorizedSamplingGrid returns a vectorized grid to easily extract
% overlapping 3-D patches from a 3-D image 
% 
% Given the size of an image, the window size (of the patches), the overlap
% of the patches, and the amount of border that is left out, this function
% computes a grid that can be easily used to extract all these patches from
% an image. The extraction of the patches then looks like 
% 
% im_patches = im(grid);
% 
% The output grid is of dimension [npixel_patch x npatches], where
% npatches is the number of patches that can be extracted from the image 
% with these settings and npixel_patch is the number of pixels in the patch
% 
% INPUTS
%  img_size     - [imh x imw x imd] size of the image to extract patches from
%  patch_size   - [ph x pw x pd] size of the patches to be extracted
%  overlapstep  - [oy x ox x od] overlap step of neighboring patches. 
%  border       - [by x bx x bd] border of image that is left out
% 
% OUTPUTS
%  grid         - [npixel_patch x npatches] the sampling grid to easily 
%                 extract patches from one 3-D image, 
%                 via: patches = im(grid);
%                 npatches is the number of patches that could be extracted
%                 and npixel_patch is the number of pixels in the patch
% 
% See also: extractPatches3D
% 
% Code adapted from [1] and [2]
% 
% @author Aline Sindel
%
% References:
% [1] S. Schulter, C. Leistner, H. Bischof. Fast and Accurate Image
%     Upscaling with Super-Resolution Forests. CVPR 2015.
% [2] R. Timofte, V. De Smet, L. van Gool. Anchored Neighborhood Regression 
%     for Fast Example-Based Super- Resolution. ICCV 2013. 

if size(img_size,2) == 3 %3-D Image
    if nargin < 4, border = [0 0 0]; end
    if nargin < 3, overlapstep = [2 2 2]; end
    grid = getVectorizedSamplingGrid3D(img_size, patch_size, overlapstep, border);
else %2-D Image
    error('Missing third dimension!');
end
end


function grid = getVectorizedSamplingGrid3D(img_size, patch_size, overlapstep, border)
% Create vectorized grid to extract 3-D patches from 3-D images

% Create sampling grid for overlapping window
index = reshape(1:prod(img_size), img_size);
grid = index(1:patch_size(1), 1:patch_size(2), 1:patch_size(3)) - 1;

% Compute offsets for grid's displacement.
offset = index(1+border(1):overlapstep(1):img_size(1)-patch_size(1)+1-border(1), ...
               1+border(2):overlapstep(2):img_size(2)-patch_size(2)+1-border(2), ...
               1+border(3):overlapstep(3):img_size(3)-patch_size(3)+1-border(3));
offset = reshape(offset, [1 1 1 numel(offset)]);

% Prepare vectorized grid - should be used as: patches = img(grid);
grid = repmat(grid, [1 1 1 numel(offset)]) + repmat(offset, [patch_size 1]);
clear index offset;
grid = reshape(grid,[size(grid,1)*size(grid,2)*size(grid,3),size(grid,4)]);
end