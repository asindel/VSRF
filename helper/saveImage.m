function saveImage(img, filenameOrig, outdir, sf, method)
% saveImage Save 3-D image as .mat, .tif, and .nii 
% INPUTS
%  img          - 3-D image to save         
%  outdir       - folder path to save image
%  sf           - super-resoluton factor
%  method       - method used for generating the image (e.g. VSRF, Tricubic)
% 
% @author Aline Sindel
%

[~,filename_base,~] = fileparts(filenameOrig);
filename = strcat(outdir, '\' , filename_base, '_', method, '_sf_', num2str(sf));
s = size(img);

%save as .mat
mat_filename = strcat(filename, '.mat');
save(mat_filename, 'img');

%save as .tif
tif_filename = strcat(filename, '.tif');
for z=1:s(3)
   imwrite(uint16 (img(:, :, z)), tif_filename, 'WriteMode', 'append',  'Compression','none');
end

%save as .nii
nifti_filename = strcat(filename, '.nii');
load(filenameOrig);

dimOrig = metadata.dim;
dimNew = s;

if any(dimOrig ~= dimNew)
    metadata.dim = dimNew;  
    myMat = metadata.mat;
    for i=1:3
        if dimOrig(i) < dimNew(i)
            myMat(i,:) = myMat(i,:)/sf;
        end
    end
    metadata.mat =  myMat;
end

metadata.fname = nifti_filename;
spm_write_vol(metadata, img);
    
end

