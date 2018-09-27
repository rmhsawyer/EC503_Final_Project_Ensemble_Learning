% Script for assignment 6.2(f)
%
% EC 503 Learning from Data
%
%% Step 1: Load data
get_adult;

SVM_classifier=svmtrain(train_data, train_label,...
        'autoscale','false','kernelcachelimit',500000,'kernel_function','linear', ...
        'boxconstraint',2,'kktviolationlevel',0.25);
    
predicted_labels = svmclassify(SVM_classifier,train_data);
CCRs=1-sum(train_label~=predicted_labels)/numel(train_label);   

predicted_labels = svmclassify(SVM_classifier,test_data);
test_CCR = 1 - sum(predicted_labels~=test_label)/length(test_label);
CFM_local = confusionmat(predicted_labels, test_label);
precision=CFM_local(2,2)/(CFM_local(2,1)+CFM_local(2,2));
recall= CFM_local(2,2)/(CFM_local(1,2)+CFM_local(2,2));
Fscore=0.5*(1/recall+ 1/precision);



% %% Step 2: Train SVM classifier
% %Creat 5-fold cross-validation partitioning
% CV = cvpartition(train_label,'k',5);
% C = zeros(1,20);
% j = -4;
% for i = 1:5
%     C(i) = 2.^j;
%     j =j+1;
% end
% clear i j
% CCRs = zeros(5,5);
% %iteration for 5-fold cross-validation partitioning
% for i = 1:5
%     train_index = CV.training(i);
%     test_index = CV.test(i);
%     %iteration for 20 different values of c
%     for j = 1:5
%         [x,~]=size(train_data(train_index,:));
%         c = C(j).*ones(x,1);
%         %Train SVM classifier
%         SVM_classifier=svmtrain(train_data(train_index,:), train_label(train_index),...
%         'autoscale','false','kernelcachelimit',500000,'kernel_function','linear', ...
%         'boxconstraint',c,'kktviolationlevel',0.25);
%         %Test SVM classifier and calculate local CCRs
%         predicted_labels = svmclassify(SVM_classifier,train_data(test_index,:));
%         CCRs(i,j)=sum(train_label(test_index)~=predicted_labels)/sum(test_index);
%     end
% end
% %Calculate average CCRs for each c
% CV_CCRs = 1 - sum(CCRs)/5;

% %% Step 4: Test OVO SVM classifier
% %start time
% test_time_strat = tic;
% count = 1;
% decision_maxtrix = zeros(length(test_labels),20*19/2);
% for i  = 1:20
%     for j = i+1:20
%         decision_maxtrix(:,count)=svmclassify(OVO_classifiers{i,j},test_data);
%         count = count+1;
%     end
% end
% predicted_labels = mode(decision_maxtrix,2);
% test_time = toc(test_time_strat);
% CFM = confusionmat(predicted_labels, test_labels);
% test_CCR = 1 - sum(predicted_labels~=test_labels)/length(test_labels);
% 
% %% Step 5: Display results
% fprintf("test CCR is %f\n",test_CCR);
% fprintf("train time is %f s, test time is %f s\n",train_time,test_time);
% disp('CFM is:');
% disp(CFM);
