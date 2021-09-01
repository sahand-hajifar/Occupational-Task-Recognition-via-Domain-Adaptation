rng('default');
close all
cd(strcat(fileparts(matlab.desktop.editor.getActiveFilename)))
addpath '..\domain-adaptation-toolbox-master'
addpath '..\domain_adaptation-master\GFK'

CD = cd;

D1 = dir([CD '\Feature1', '\*.csv']);
filenames1 = {D1(:).name}.';
empatica_data = cell(length(D1),1);
for ii = 1:length(D1)
    % Create the full file name and partial filename
    fullname = [CD '\Feature1\' D1(ii).name];
    % Read in the data
    empatica_data{ii} = readtable(fullname);
end

D2 = dir([CD '\Feature2', '\*.csv']);
filenames2 = {D2(:).name}.';
maxim_data = cell(length(D2),1);
for ii = 1:length(D2)
    % Create the full file name and partial filename
    fullname = [CD '\Feature2\' D2(ii).name];
    % Read in the data
    maxim_data{ii} = readtable(fullname);
end

subject_ids = 1:10;

F1_DA = cell(1,length(subject_ids));
MeanF1_DA = zeros(1,length(subject_ids));
Acc_DA = zeros(1,length(subject_ids));

for i = 1:length(subject_ids)
    
    train_data = empatica_data(Selected_Source{i});
    train_data = cat(1,train_data{:});
    min_sample = min(groupcounts(train_data.label));
    [~,~,X] = unique(train_data.label);
    C = accumarray(X,1:size(train_data,1),[],@(r){train_data(r,:)});
    C = cellfun(@(x) x(randsample(size(x,1),min_sample),:),C,'uniformoutput',false);
    train_data = cat(1,C{:});
    
    test_data = maxim_data{subject_ids(i)};
    
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
    %Mdl_DA = fitcecoc(train_ft,train_labels,'OptimizeHyperparameters','auto',...
    %     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    %     'expected-improvement-plus'));
    %Mdl_DA = fitcknn(train_ft,train_labels,'OptimizeHyperparameters','auto',...
    %             'HyperparameterOptimizationOptions',...
    %             struct('AcquisitionFunctionName','expected-improvement-plus'));
    
    
    test_pred_DA = predict(Mdl_DA, test_ft);
    
    [g,gN] = grp2idx(Labels);
    test_labels = gN(test_labels);
    test_pred_DA = gN(test_pred_DA);
    
    C_DA = confusionmat(test_labels,test_pred_DA);
    figure()
    confusionchart(test_labels,test_pred_DA)
    title(['DA - Participant ', int2str(i)])
    saveas(gcf,['P',int2str(i),'_DA.png'])
    
    F1_DA{i} = (getF1(C_DA))';
    MeanF1_DA(i) = mean(F1_DA{i});
    Acc_DA(i) = (sum(diag(C_DA)))/(sum(sum(C_DA)));
end

clearvars -except MeanF1_DA F1_DA Acc_DA num_source

mean(Acc_DA)