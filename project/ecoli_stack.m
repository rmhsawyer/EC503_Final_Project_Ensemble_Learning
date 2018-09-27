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

train_labels = train_data(:,8);
train_data(:,8) = [];

test_labels = test_data(:,8);
test_data(:,8) = [];
% get_adult();
% load train_data.mat
% load train_label.mat
% load test_data.mat
% load test_label.mat

test_label = test_labels;
train_label = train_labels;


% Perofrmance of single learner
% 1. GDA
% 2. knn (NumNeighbors = 30)
% 4. DT
% 5. SVM (rbf)
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

% weak_learner=fitcnb(train_data,train_label);
% predicted5=predict(weak_learner, train_data);
% CCR5_1 = 1- sum(predicted5 ~= train_label)/length(predicted5);
% predicted6=predict(weak_learner, test_data);
% CCR5 = 1- sum(predicted6 ~= test_label)/length(predicted6);

weak_learner=fitctree(train_data,train_label);
predicted7=predict(weak_learner, train_data);
CCR6_1 = 1- sum(predicted7 ~= train_label)/length(predicted7);
predicted8=predict(weak_learner, test_data);
CCR6 = 1- sum(predicted8 ~= test_label)/length(predicted8);




% 1. GDA
% 2. knn (NumNeighbors = 30)
% 3. DT
% 4. SVM (rbf)


new_train_data = [train_data,predicted1,predicted3,predicted7,predicted_labels];
new_test_data = [test_data,predicted2,predicted4,predicted8,test_predicted_labels];
new_regression=fitctree(new_train_data,train_labels);
new_train_predict = predict(new_regression, new_train_data);
CCR_stacking_train = 1 -sum(new_train_predict~= train_label)/length(new_train_predict);

new_test_predict = predict(new_regression, new_test_data);
CCR_stacking_test = 1 -sum(new_test_predict~= test_label)/length(new_test_predict);

% CCR we recorded
test_ada_CCR = [0.8416;0.7426;0.8218;0.7822;0.8319];
train_ada_CCR = [0.8766;0.8170;0.9234;0.8340;0.9489];
figure(2)
hold on
bar(test_ada_CCR)
legend('test CCR','training CCR')
axe_labels = {'GDA','knn', 'Decision Tree','SVM','Stacking'};
set(gca,'xlim',[0,6], 'xTick',1:5, 'xticklabel',axe_labels);
xtickangle(45);
xlabel('Learning Algorithm')
ylabel('CCR')
title('CCR against different learning agorithms')
hold off


