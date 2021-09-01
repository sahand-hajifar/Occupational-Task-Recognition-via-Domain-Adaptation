%%% Change current directory to the Maxim data directory
cd(strcat(fileparts(matlab.desktop.editor.getActiveFilename),'\Data'))
%%% Extract the information related to Maxim csv files
files = dir(strcat(fileparts(matlab.desktop.editor.getActiveFilename),'\Data\*.csv'));

%%% Read all of the acc data and store them in DATA
DATA = cell(size(files,1),1);
for i = 1:size(files,1)
    DATA{i} = readmatrix(files(i).name);
end

%%% Extract the data based on the experiment day (look into files.name)
Data = DATA{15};

%%% Enter the start and end of the experiment (unix time) based on annotation info.
Start = 1604502139000;
End = 1604506260000;

Data1 = Data(Data(:,1)>Start&Data(:,1)<End,:);

%%% Save Data1 in './Extracted_Data' folder to be used in second step
%%% (removing missing parts and interpolations). I name the files as
%%% 'ACCX', where X is the number of Subject
 
