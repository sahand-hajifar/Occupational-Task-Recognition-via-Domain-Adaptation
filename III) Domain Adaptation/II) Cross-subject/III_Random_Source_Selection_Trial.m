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

subject_ids = 1:10;

F1 = cell(1,length(subject_ids));
MeanF1 = zeros(1,length(subject_ids));
Acc = zeros(1,length(subject_ids));

for i = 1:length(subject_ids)
    
    train_data = empatica_data(randsample(setdiff(subject_ids,i),num_source));
    train_data = cat(1,train_data{:});
    min_sample = min(groupcounts(train_data.label));
    [~,~,X] = unique(train_data.label);
    C = accumarray(X,1:size(train_data,1),[],@(r){train_data(r,:)});
    C = cellfun(@(x) x(randsample(size(x,1),min_sample),:),C,'uniformoutput',false);
    train_data = cat(1,C{:});
    
    test_data = empatica_data{subject_ids(i)};
    
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
    
    Mdl = fitcecoc(train_features,train_labels);
    
    [g,gN] = grp2idx(Labels);
    test_labels = gN(test_labels);
    test_pred = predict(Mdl, test_features);
    test_pred = gN(test_pred);
    
    C = confusionmat(test_labels,test_pred);
    figure()
    confusionchart(test_labels,test_pred)
    title(['Without DA - Participant ', int2str(i)])
    saveas(gcf,['P',int2str(i),'.png'])
    
    F1{i} = (getF1(C))';
    MeanF1(i) = mean(F1{i});
    Acc(i) = (sum(diag(C)))/(sum(sum(C)));
end

F1 = cat(1,F1{:});
F1_DA = cat(1,F1_DA{:});

%%% Taks specific F1 score in the table for cross-subject case
Final_Results = zeros(size(F1,1)+size(F1_DA,1),size(F1,2));
Final_Results(1:2:end,:) = F1;
Final_Results(2:2:end,:) = F1_DA;

%%% Overall F1 score results in the table for cross-subject case
Final_Results2 = zeros(size(F1,1)+size(F1_DA,1),1);
Final_Results2(1:2:end,1) = MeanF1;
Final_Results2(2:2:end,1) = MeanF1_DA;

%%% Overall accuracy results in the table for cross-subject case
Final_Results3 = zeros(size(Acc,2)+size(Acc_DA,2),1);
Final_Results3(1:2:end,1) = Acc;
Final_Results3(2:2:end,1) = Acc_DA;

mean(Acc)