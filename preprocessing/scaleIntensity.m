% scaleIntensity Scale intensity of Kirby data from float32 to int16. For 
% other datasets you might to have to adapt the divisor. Scale the data to
% have maxInt as the maximum value (=32767).
%
% @author Aline Sindel
%

myFolder = uigetdir(pwd,'Select folder of Nifti-Files to convert');
myPath = strcat(myFolder, '\*.mat');
mkdir(strcat(myFolder,'\intensityScaled\'));
files=dir(myPath);

for k=1:length(files)
    filename = files(k).name;    
    myFile = strcat(myFolder,'\', filename);
    mat_filename = strcat(myFolder,'\intensityScaled\', filename);
    
    load(myFile);

    %scale intensity (Kirby data is float32 - scale to int16)
    data = data*double(intmax('int16'))/2.408321500000000e+06; 
    
    %save as .mat    
    save(mat_filename, 'data', 'metadata');  
    
end
    
