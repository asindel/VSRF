function imlist = getInputFiles(inputfolders)
% create a list of all mat-files in the input folders
% 
% @author Aline Sindel
%

imlist = {};
for i=1:size(inputfolders,2)
    filesinfolder = strcat(inputfolders{i}, '\*.mat');
    files=dir(filesinfolder);
    for k=1:length(files)       
        imlist = [imlist , fullfile(files(k).folder, files(k).name)];                
    end
end
end