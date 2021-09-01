rng('default');
close all
cd(strcat(fileparts(matlab.desktop.editor.getActiveFilename)))
addpath '..\domain-adaptation-toolbox-master'
addpath '..\domain_adaptation-master\GFK'

CD = cd;

D1 = dir([CD '\Feature1', '\*.csv']);
filenames1 = {D1(:).name}.';
separate_data = cell(length(D1),1);
for ii = 1:length(D1)
    % Create the full file name and partial filename
    fullname = [CD '\Feature1\' D1(ii).name];
    % Read in the data
    separate_data{ii} = readtable(fullname);
end


D2 = dir([CD '\Feature2_Mixed', '\*.csv']);
filenames2 = {D2(:).name}.';
mixed_data = cell(length(D2),1);
for ii = 1:length(D2)
    % Create the full file name and partial filename
    fullname = [CD '\Feature2_Mixed\' D2(ii).name];
    % Read in the data
    mixed_data{ii} = readtable(fullname);
end


train_data = vertcat(separate_data{:});
train_data(strcmp(train_data.label, 'type'),:) = [];
min_sample = min(groupcounts(train_data.label));
[~,~,X] = unique(train_data.label);
C = accumarray(X,1:size(train_data,1),[],@(r){train_data(r,:)});
C = cellfun(@(x) x(randsample(size(x,1),min_sample),:),C,'uniformoutput',false);
train_data = cat(1,C{:});

test_data = vertcat(mixed_data{:});
test_data.label(strcmp(test_data.label,'climb-ladder')|strcmp(test_data.label,'descend-ladder')) = {'ladder'};
test_data.label(strcmp(test_data.label,'hoist-on-ladder')) = {'hoist'};

train_features = (table2array(train_data(:,2:end-1)));
[train_features,mu,sigma] = zscore(train_features);
train_labels = categorical(table2array(train_data(:,1)));

test_features = (table2array(test_data(:,2:end-1)));
mu = repmat(mu,size(test_features,1),1);
sigma = repmat(sigma,size(test_features,1),1);
test_features = (test_features-mu)./sigma;
test_labels = categorical(table2array(test_data(:,1)));

Labels = [train_labels; test_labels];
labels = grp2idx(Labels);

train_labels = labels(1:size(train_labels,1));
test_labels = labels((size(train_labels,1)+1):end);

ftAll = [train_features; test_features];
maSrc = false(size(ftAll,1),1);
maSrc(1:size(train_features,1)) = true;
target = labels(1:size(train_features,1));
maLabeled = maSrc;

[ftAllNew,transMdl] = ftTrans_gfk(ftAll,maSrc,target,maLabeled);

train_ft = ftAllNew(1:size(train_labels,1),:);
test_ft = ftAllNew((size(train_labels,1)+1):end,:);

Mdl_DA = fitcecoc(train_ft,train_labels);

test_pred_DA = predict(Mdl_DA, test_ft);

[g,gN] = grp2idx(Labels);
test_labels = gN(test_labels);
test_pred_DA = gN(test_pred_DA);

C_DA = confusionmat(test_labels,test_pred_DA);
figure()
confusionchart(test_labels,test_pred_DA)
title('Domain Adaptation')
saveas(gcf,'DA.png')

F1_DA = (getF1(C_DA))';
MeanF1_DA = mean(F1_DA);
Acc_DA = (sum(diag(C_DA)))/(sum(sum(C_DA)));

Mdl = fitcecoc(train_features,train_labels);

test_pred = predict(Mdl, test_features);
test_pred = gN(test_pred);

C = confusionmat(test_labels,test_pred);
figure()
confusionchart(test_labels,test_pred)
title('Without DA')
saveas(gcf,'No_DA.png')

F1 = (getF1(C))';
MeanF1 = mean(F1);
Acc = (sum(diag(C)))/(sum(sum(C)));