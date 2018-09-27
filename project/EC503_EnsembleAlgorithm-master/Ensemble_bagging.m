function [test_set_predicted_Y] = Ensemble_bagging(train_set_X, train_set_Y, test_set_X, n_prime_percent, class_total)
% Three different weak learners // QDA, Decision Trees, and SVM
% Input: data 
% Output: Label 

% EC 503: Learning from Data
% Instructor: Prakash Ishwar
% Assignment 8, Problem 8.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
train_size = size(train_set_X,1);
test_size = size(test_set_X,1);
test_mat = zeros(test_size,3); 

select_train_1 = datasample(1:train_size,round(train_size * n_prime_percent));
train_QDA = train_set_X(select_train_1,:);
test_QDA = train_set_Y(select_train_1,:);
QDA_Classfier = fitcdiscr(train_QDA,test_QDA,'DiscrimType','pseudoLinear'); 

select_train_2 = datasample(1:train_size,round(train_size * n_prime_percent));
train_DT = train_set_X(select_train_2,:);
test_DT = train_set_Y(select_train_2,:);
DT_Classfier = fitctree(train_DT,test_DT); 

select_train_3 = datasample(1:train_size,round(train_size * n_prime_percent));
train_SVM = train_set_X(select_train_3,:);
test_SVM = train_set_Y(select_train_3,:);

if class_total == 2
    SVM_Classfier = svmtrain(train_SVM,test_SVM,'Autoscale','false','BoxConstraint',...
        1,'kernelcachelimit',1000000,'kernel_function','linear','kktviolationlevel',0.25);
    test_mat(:,3) = svmclassify(SVM_Classfier,test_set_X);
else
    mat = zeros(test_size,class_total);
    for m = 1:(class_total-1)
        k = m + 1; 
        for n = k:class_total
            tempIndex_m = ismember(train_set_Y,m);
            tempIndex_n = ismember(train_set_Y,n);
            svm_training_X = [train_set_X(tempIndex_m,:);train_set_X(tempIndex_n,:)];
            svm_training_Y = [train_set_Y(tempIndex_m,:);train_set_Y(tempIndex_n,:)];
            svmStructure = svmtrain(svm_training_X,svm_training_Y,'Autoscale','false',...
                'BoxConstraint',1,'kernelcachelimit',1000000,'kernel_function','linear');
            test_predicted = svmclassify(svmStructure,test_set_X);
            for item = 1:test_size
                result = test_predicted(item); 
                mat(item,result) = mat(item,result) + 1; 
            end 
        end
    end 
    test_predict = zeros(test_size,1);
    for i = 1:test_size
        [M,I]= max(mat(i,:));
        test_predict(i) = I;
    end 
    test_mat(:,3) = test_predict; 
end 

test_mat(:,1) = predict(QDA_Classfier,test_set_X); 
test_mat(:,2) = predict(DT_Classfier,test_set_X); 
test_set_predicted_Y = mode(test_mat,2); 






