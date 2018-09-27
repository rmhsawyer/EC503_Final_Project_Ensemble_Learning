ecoli = readtable('dataset/Ecoli_Data/ecoli.data.txt');
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





