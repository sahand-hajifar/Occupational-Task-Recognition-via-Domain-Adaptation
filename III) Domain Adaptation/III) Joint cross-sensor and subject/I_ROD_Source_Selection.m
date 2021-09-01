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
Selected_Source = cell(1,length(subject_ids));
Selected_Source_ROD = cell(1,length(subject_ids));
num_source = 5;

for i = 1:length(subject_ids)
    
    source_subject_ids = setdiff(subject_ids,i);
    ROD_res = zeros(1,length(source_subject_ids));
    
    for j = 1:length(source_subject_ids)
        
        train_data = empatica_data{source_subject_ids(j)};
        min_sample = min(groupcounts(train_data.label));
        [~,~,X] = unique(train_data.label);
        C = accumarray(X,1:size(train_data,1),[],@(r){train_data(r,:)});
        C = cellfun(@(x) x(randsample(size(x,1),min_sample),:),C,'uniformoutput',false);
        train_data = cat(1,C{:});
        
        train_features = (table2array(train_data(:,2:end-1)));
        [train_features,mu,sigma] = zscore(train_features);
        train_labels = categorical(table2array(train_data(:,1)));
        
        test_data = maxim_data{subject_ids(i)};
        
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
        
        ROD_res(j) = rank_of_domain(ftAll,maSrc,target,maLabeled);
    end
    [Min,Ind] = mink(ROD_res, num_source);
    Selected_Source{i} = source_subject_ids(Ind);
    Selected_Source_ROD{i} = Min;
    
end

clearvars -except Selected_Source Selected_Source_ROD num_source