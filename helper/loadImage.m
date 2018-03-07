function [img, voxelspacing] = loadImage(filename)
% loadImage Load .mat file with image data from the file path defined by 
% filename
% 
% @author Aline Sindel
%

myImg = load(filename);
img = myImg.data;
voxelspacing = getVoxelspacing(myImg.metadata.mat);
clear myImg;

end

function voxelspacing = getVoxelspacing(trafoMat)
% extract the voxelspacing from the transformation matrix of the metadata
voxelspacing = zeros(1,3);
for i=1:3
    voxelspacing(i) = trafoMat(i,i);
end

end