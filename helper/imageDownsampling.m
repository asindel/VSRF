function img = imageDownsampling(img, factor, method)
% imageDownsampling function to perform downsampling of a 2-D or 3-D image
% (in-place)
%
% INPUTS
%  img      - image to be downsampled
%  factor   - downsampling factor
%  method   - defines the methods that is used for downsampling
%              2-D: bicubic, bilinear, nearest, gaussian
%              3-D: tricubic, trilinear, nearest3, gaussian3,
%                   tricubic*, trilinear* (*=without Antialiasing filter)
% 
% @author Aline Sindel
%

if strcmp(method, 'bicubic') || strcmp(method, 'bilinear') || strcmp(method, 'nearest')
    img = imresize(img,factor,method);
elseif strcmp(method, 'tricubic') || strcmp(method, 'trilinear')
    method = method(4:end);    
    img = imresize3(img,factor,method);
elseif strcmp(method, 'tricubic*') || strcmp(method, 'trilinear*')
    method = method(4:end-1);
    img = imresize3(img,factor,method,'Antialiasing',false);
elseif strcmp(method, 'nearest3')
    img = imresize3(img,factor,'nearest');
elseif strcmp(method, 'gaussian')
    img = imgaussfilt3(img,1);
    img = imresize(img,factor,'nearest');
elseif strcmp(method, 'gaussian3')
    img = imgaussfilt3(img,1);
    img = imresize3(img,factor,'nearest'); 
else
    error(message('imageDownsampling:InvalidMethod', deblank( method )));
end

end