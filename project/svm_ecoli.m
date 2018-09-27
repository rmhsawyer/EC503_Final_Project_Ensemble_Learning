ecoli = readtable('ecoli.data.txt');
% Check missing values // No missing values 
sum(sum(ismissing(ecoli))) == 0;
% First Column is the primary key
numel(unique(ecoli(:,1))) == size(ecoli,1);

% Delete the first Col 
data = table2array(ecoli(:,2:end-1));
place_holder = zeros(size(ecoli,1),1);
data = [data,place_holder];
% Rename the last column // Categorical as num 
list_labels = unique(ecoli(:,end));
for i = 1:numel(list_labels)
    index = ismember(ecoli(:,end),list_labels(i,1));
    data(index,end) = i;
end 

stream = RandStream('mt19937ar','Seed',88); 
RandStream.setGlobalStream(stream);
randomIndex = randperm(336);
train_index = randomIndex(1:235);
test_index = randomIndex(236:336);

% 70% train and 30% test
train_data = data(train_index,:);
test_data = data(test_index,:);

clear ecoli i index randomIndex stream ans

train_labels = train_data(:,8);
train_data(:,8) = [];

test_labels = test_data(:,8);
test_data(:,8) = [];
% SVM_classifier=svmtrain(train_data, train_label,...
%         'autoscale','false','kernelcachelimit',500000,'kernel_function','linear', ...
%         'boxconstraint',2,'kktviolationlevel',0.25);
    
OVO_classifiers = cell(8,8);
%Loop over each classifier(1: 20)
for i = 1:8
    %loop over i+1 to 20
    for j = i+1:8 
        %Get index for class 1 to 20 in training set
        label1 = find(train_labels == i);
        label2 = find (train_labels == j);
        indexs = [label1;label2];
        new_train_data = train_data(indexs,:);
        new_train_labels = train_labels(indexs,:);
        %Get index for class 1 to 20 in test set
        label1 = find(test_labels == i);
        label2 = find (test_labels == j);
        indexs = [label1;label2];
        new_test_data = test_data(indexs,:);
        new_test_labels = test_labels(indexs,:);
        %Train SVM classifier
        OVO_classifiers{i,j}=svmtrain(new_train_data,new_train_labels,'autoscale','false','kernelcachelimit',500000,'kernel_function','linear','boxconstraint',2.^4);
    end
end

count = 1;
decision_maxtrix = zeros(length(train_labels),8*7/2);
for i  = 1:8
    for j = i+1:8
        decision_maxtrix(:,count)=svmclassify(OVO_classifiers{i,j},train_data);
        count = count+1;
    end
end
predicted_labels = mode(decision_maxtrix,2);
CFM = confusionmat(predicted_labels, train_labels);
train_CCR = 1 - sum(predicted_labels~=train_labels)/length(train_labels);

count = 1;
decision_maxtrix = zeros(length(test_labels),8*7/2);
for i  = 1:8
    for j = i+1:8
        decision_maxtrix(:,count)=svmclassify(OVO_classifiers{i,j},test_data);
        count = count+1;
    end
end
test_predicted_labels = mode(decision_maxtrix,2);
CFM = confusionmat(test_predicted_labels, test_labels);
test_CCR = 1 - sum(test_predicted_labels~=test_labels)/length(test_predicted_labels);

% count = 1;
% decision_maxtrix = zeros(length(test_labels),8*7/2);
% for i  = 1:8
%     for j = i+1:8
%         decision_maxtrix(:,count)=svmclassify(OVO_classifiers{i,j},test_data);
%         count = count+1;
%     end
% end
% predicted_labels = mode(decision_maxtrix,2);
% CFM = confusionmat(predicted_labels, test_labels);
% test_CCR = 1 - sum(predicted_labels~=test_labels)/length(test_labels);
