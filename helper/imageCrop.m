function imCrop = imageCrop( im , x0, x1, y0, y1, z0, z1)
%imageCrop crop 3-D image to the indices defined as parameters
% 
% @author Aline Sindel
%
imCrop = im(x0:x1,y0:y1,z0:z1);
end