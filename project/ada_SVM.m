% load data
% load train.mat
% load train_label.mat
% load('test.mat')
% load('test_label.mat')
train_data(:,11:14) = [];
test_data(:,11:14) = [];
% modify the label to be +1 and -1
trlabel = train_label + (train_label == 1) - 1;

% seperate training data into training set and testing set rst 70% and 30%
n= length(train_label);
randomIndex = randperm(n);
train_index = randomIndex(1:0.7*n);
test_index = randomIndex(0.7*n:n);
train_data1 =train_data(train_index,:);
trlabel1 = trlabel(train_index);
test_data1 =train_data(test_index,:);
telabel1 = trlabel(test_index,:);

% test for the weak_leaner performance
weak_learner=fitcdiscr(train_data,train_label);
predicted=predict(weak_learner, test_data);
CCR3 = 1- sum(predicted ~= test_label)/length(predicted);

% Adaboosting
% Weak learner: GDA
% number of iterations
M = 10;

sample = [train_data1 trlabel1];
% number of samples
N = size(train_data1, 1);
% inital the weight to 1/n for each sample
W = ones(N, 1) ./ N;


% inital the hypothesis
h_hat = zeros(N, M);
% inital the summary error for each iteration
beta = zeros(M, 1);
% inital the error for each iteration
err = zeros(M,1);
% inital the prediction
train_predict = zeros(size(train_data1, 1), M);
test_predict  = zeros(length(telabel1), M);
% intial CCR
train_ada_CCR = zeros(M, 1);
test_ada_CCR  = zeros(M, 1);
tmp_CCR = zeros(M,1);
for iter = 1 : M
   
    % initial the training set for each iteration
    train_set = datasample(sample,size(sample,1),'Replace',true,'Weights',W); 

    % seperate the training set in to sample and label
    X = train_set(:, 1 : size(train_set,2)-1);
    Y = train_set(:, size(train_set,2));

    % apply GDA
    nb_mdl = svmtrain(X, Y,...
        'autoscale','false','kernelcachelimit',500000,'kernel_function','linear','boxconstraint',2,'kktviolationlevel',0.25);
    h_hat(:, iter) = svmclassify(nb_mdl, X);

    % weighted sum error for misclassified points
    err(iter) = sum(W(h_hat(:, iter) ~= Y));
    
    % Hypothesis weight
    beta(iter) = 1 / 2 * log((1 - err(iter)) / err(iter));
    
    % Update weights
    W = W .* exp((-1) .* Y .* beta(iter) .* h_hat(:, iter));
    W = W ./ sum(W); % normalize the weight
    
    % final vote
    tmp_predict = svmclassify(nb_mdl, X);
    train_predict(:,iter) = svmclassify(nb_mdl, train_data1);
    test_predict(:,iter)  = svmclassify(nb_mdl, test_data1);
    
    % calculating Adaboosting CCR for each iteration
    tmp_CCR(iter) = 1- sum(tmp_predict ~= Y) / length(Y);
    train_ada       = sign(sum(train_predict * beta, 2));
    train_ada_CCR(iter) = 1- sum(train_ada ~= trlabel1) / length(trlabel1);
    test_ada       = sign(sum(test_predict * beta, 2));
    test_ada_CCR(iter)  = 1- sum(test_ada ~= telabel1) / length(telabel1);
    
end

% compare CCR for each iteration
train_CCR = zeros(M ,1);
for i = 1:M
    train_CCR(i) = 1- sum(train_predict(:,i) ~= trlabel1)/length(trlabel1);
end
test_CCR = zeros(M ,1);
for i = 1:M
    test_CCR(i) = 1- sum(test_predict(:,i) ~= telabel1)/length(telabel1);
end

clear  d gda_mdl h_hat i iter  max_w min_w N p predicted sample t
clear test_ada_QDA test_data test_label 
clear train_ada_QDA train_data train_label train_set
clear telabel trlabel weak_learner X Y

% Adaboosting CCR for each iteration
figure
hold on
plot(1:M, train_ada_CCR)
plot(1:M, test_ada_CCR)
plot(1:M, train_CCR)
plot(1:M, test_CCR)
plot(1:M, tmp_CCR)
legend('Train Adaboosting CCR','Test Adaboosting CCR',...
    'Train CCR of each iteration','Test CCR of each iteration',...
    'Train CCR for dataset at each iteration')
xlabel('Number of Iteration')
ylabel('CCR')
title('Adaboosting with Naive Bayes as Weak Learner')
hold off
[max_v,i] = sort(test_ada_CCR,'descend');
best_CCR = [max_v i];