% saveAsmat_FOV convert MR volumes (.nii) to .mat files and crop them
%
% @author Aline Sindel
%

addpath('../helper/');
myFolder = uigetdir(pwd,'Select folder of Nifti-Files to convert');
myPath = strcat(myFolder, '\*.nii');
files=dir(myPath);

for k=1:length(files)
    filename = files(k).name;    
    myFile = strcat(myFolder,'\', filename);
    
    %get id and acquisition no.
    [~,filename_base,~] = fileparts(filename);    
    
    %read Nifti file
    metadata = spm_vol(myFile);
    [data] = spm_read_vols(metadata);
    
    %following settings are optional:
    
    %rotation etc. specified for Kirby data, change for your data
    data = permute(data,[3 2 1]);
    data = rot90(data,2);
    
    myMat = metadata.mat;
    matNew = eye(4,4);
    matNew(1,1) = myMat(3,3); matNew(1,4) = myMat(3,4);
    matNew(2,2) = myMat(2,2); matNew(2,4) = myMat(2,4);
    matNew(3,3) = myMat(1,1); matNew(3,4) = myMat(1,4);
    metadata.mat = matNew;  
    
    %crop data to a field of view (240 x 256 x 170 in case of Kirby)
    data = imageCrop( data , 17, 256, 1, 256, 1, 170); 
    
    dimNew = size(data);
    metadata.dim = dimNew;
    
    mat_filename = strcat(myFolder,'\', filename_base, '_FOV_', num2str(dimNew(1)), '_', num2str(dimNew(2)), '_', num2str(dimNew(3)), '.mat'); 
      
    save(mat_filename, 'metadata', 'data');
end

