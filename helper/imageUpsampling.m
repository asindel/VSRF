function img = imageUpsampling(img, factor, method)
% imageUpsampling function to perform upsampling of a 2-D or 3-D image 
% (in-place)
%
% INPUTS
%  img      - image to be upsampled
%  factor   - upsampling factor
%  method   - defines the methods that is used for upsampling
%              2-D: bicubic, bilinear, nearest
%              3-D: tricubic, trilinear, nearest3
% 
% @author Aline Sindel
%

if strcmp(method, 'bicubic') || strcmp(method, 'bilinear') || strcmp(method, 'nearest')
    img = imresize(img,factor,method);
elseif strcmp(method, 'tricubic') || strcmp(method, 'trilinear')
    method = method(4:end);
    img = imresize3(img,factor,method);
elseif strcmp(method, 'nearest3')
    img = imresize3(img,factor,'nearest');
else
    error(message('imageUpsampling:InvalidMethod', deblank( method )));
end

end