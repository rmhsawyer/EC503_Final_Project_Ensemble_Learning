ecoli = readtable('ecoli.data.txt');
% Check missing values // No missing values 
sum(sum(ismissing(ecoli))) == 1;
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

weak_learner=fitcdiscr(train_data,train_label,'discrimType','pseudoLinear');
predicted1=predict(weak_learner, train_data);
CCR3_1 = 1- sum(predicted1 ~= train_label)/length(predicted1);
predicted2=predict(weak_learner, test_data);
CCR3 = 1- sum(predicted2 ~= test_label)/length(predicted2);

weak_learner=fitcknn(train_data,train_label,'NumNeighbors',30);
predicted3=predict(weak_learner, train_data);
CCR4_1 = 1- sum(predicted3 ~= train_label)/length(predicted3);
predicted4=predict(weak_learner, test_data);
CCR4 = 1- sum(predicted4 ~= test_label)/length(predicted4);

weak_learner=fitcnb(train_data,train_label);
predicted5=predict(weak_learner, train_data);
CCR5_1 = 1- sum(predicted5 ~= train_label)/length(predicted5);
predicted6=predict(weak_learner, test_data);
CCR5 = 1- sum(predicted6 ~= test_label)/length(predicted6);

weak_learner=fitctree(train_data,train_label);
predicted7=predict(weak_learner, train_data);
CCR6_1 = 1- sum(predicted7 ~= train_label)/length(predicted7);
predicted8=predict(weak_learner, test_data);
CCR6 = 1- sum(predicted8 ~= test_label)/length(predicted8);

weak_learner=fitcsvm(train_data,train_label,'KernelFunction','rbf');
predicted=predict(weak_learner, train_data);
CCR7_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR7 = 1- sum(predicted ~= test_label)/length(predicted);



new_train_data = [train_data,predict,predicted2,predicted4];
new_test_data = [test_data,predict1,predicted2,predicted4]];
new_regression=fitctree(new_train_data,train_label);
new_train_predict = predict(new_regression, new_train_data);
CCR_stacking_train = 1 -sum(new_train_predict~= train_label)/length(new_train_predict);