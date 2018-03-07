function imCrop = imageModcrop( im, modfactor, vol)
% imageModcrop Crops a 2-D or 3-D image to be divideable without remainder.
%
% Given an image im and a modulo-factor modfactor, this function crops the
% image such that it is exactly dividable by modfactor, i.e., without any
% remainder. 
% 
% INPUTS
%  im           - [imh x imw] or [imh x imw x imd] a 2-D or 3-D image
%  modfactor    - the modulo factor
%  vol          - [0] or [1] defines if the z-direction is cropped too,
%                 if parameter is not set, z-direction is not considered
% 
% OUTPUTS
%  imCrop       - the cropped image
% 
% Code adapted from [1] and [2]
% 
% References:
% [1] S. Schulter, C. Leistner, H. Bischof. Fast and Accurate Image
%     Upscaling with Super-Resolution Forests. CVPR 2015.
% [2] R. Timofte, V. De Smet, L. van Gool. Anchored Neighborhood Regression 
% for Fast Example-Based Super- Resolution. ICCV 2013. 

[imh,imw,imd]=size(im); 
imh=imh-mod(imh,modfactor); 
imw=imw-mod(imw,modfactor);
if nargin>2 && vol==1 %crop all three directions
    imd=imd-mod(imd,modfactor);
    imCrop = im(1:imh,1:imw,1:imd);
else %crop only x- and y-direction
    imCrop = im(1:imh,1:imw,:);
end
end