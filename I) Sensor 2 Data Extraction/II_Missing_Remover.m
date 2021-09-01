%%% Change current directory to the acc data directory
cd(strcat(fileparts(matlab.desktop.editor.getActiveFilename),'\Extracted_Data'))
%%% Extract the information related to Maxim acc. csv files
files = dir(strcat(fileparts(matlab.desktop.editor.getActiveFilename),'\Extracted_Data\*.mat'));

for i = 1:size(files,1)

load(files(i).name)

diff = Data1(2:end,1)-Data1(1:(end-1),1);

temp = find(diff>1000);
Remove = zeros(size(temp,1),2);
Remove(:,1) = Data1(temp,1);
Remove(:,2) = Data1(temp+1,1);

DATA = Data1(:,[1 8 9 10]);

x = DATA(:,1);
v = DATA(:,2:4);

xq = (x(1):(1000/32):x(end))';

vq = interp1(x,v,xq);

for j = 1:size(Remove,1)
    temp_rem = find(xq>Remove(j,1)&xq<Remove(j,2));
    vq(temp_rem,:) = NaN;
end

ACC = [x(1)/1000 x(1)/1000 x(1)/1000;32 32 32;64*vq];
%%% First row of the ACC is the start time, second row shows the sampling
%%% rate, and from third row we have the acc data
writematrix(ACC,[files(i).name(1:5),'.csv']) 
end


