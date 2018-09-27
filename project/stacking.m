 
% get_adult();
load train_data.mat
load train_label.mat
load test_data.mat
load test_label.mat

%Make data label -1 and 1
n= length(train_label);
for i = 1:n
    if(train_label(i)== 0)
        train_label(i) = -1;
    end
end

%Make data label -1 and 1
n= length(test_label);
for i = 1:n
    if(test_label(i)== 0)
        test_label(i) = -1;
    end
end

% Perofrmance of single learner
% 1. GDA
% 2. knn (NumNeighbors = 30)
% 3. Naive Bayes
% 4. DT
% 5. SVM (rbf)
% weak_learner=fitcdiscr(train_data,train_label,'discrimType','pseudoLinear');
% predicted=predict(weak_learner, train_data);
% CCR3_1 = 1- sum(predicted ~= train_label)/length(predicted);
% predicted=predict(weak_learner, test_data);
% CCR3 = 1- sum(predicted ~= test_label)/length(predicted);
% 
% weak_learner=fitcknn(train_data,train_label,'NumNeighbors',30);
% predicted=predict(weak_learner, train_data);
% CCR4_1 = 1- sum(predicted ~= train_label)/length(predicted);
% predicted=predict(weak_learner, test_data);
% CCR4 = 1- sum(predicted ~= test_label)/length(predicted);
% 
% weak_learner=fitcnb(train_data,train_label);
% predicted=predict(weak_learner, train_data);
% CCR5_1 = 1- sum(predicted ~= train_label)/length(predicted);
% predicted=predict(weak_learner, test_data);
% CCR5 = 1- sum(predicted ~= test_label)/length(predicted);
% 
% weak_learner=fitctree(train_data,train_label);
% predicted=predict(weak_learner, train_data);
% CCR6_1 = 1- sum(predicted ~= train_label)/length(predicted);
% predicted=predict(weak_learner, test_data);
% CCR6 = 1- sum(predicted ~= test_label)/length(predicted);
% 
% weak_learner=svmtrain(train_data, train_label,...
%         'autoscale','false','kernelcachelimit',500000,'kernel_function','linear', ...
%         'boxconstraint',2,'kktviolationlevel',0.25);
% predicted=svmclassify(weak_learner, train_data);
% CCR7_1 = 1- sum(predicted ~= train_label)/length(predicted);
% predicted=svmclassify(weak_learner, test_data);
% CCR7 = 1- sum(predicted ~= test_label)/length(predicted);
Xtrain = train_data;
Ytrain = train_label;

Xtest = test_data;
Ytest = test_label;
%Creat 5 partitioning

Xtrain1=train_data(1:6512,:);
Ytrain1 =train_label(1:6512);

Xtrain2=train_data(6513:13024,:);
Ytrain2 =train_label(6513:13024);

Xtrain3 = train_data(13025:19536,:);
Ytrain3 =train_label(13025:19536);

Xtrain4 = train_data(19537:26048,:);
Ytrain4 =train_label(19537:26048);

Xtrain5 = train_data(26049:32560,:);
Ytrain5 =train_label(26049:32560);
% Choosen Weak classifiers as stacking method:
% 1. GDA
% 2. knn (NumNeighbors = 30)
% 3. Naive Bayes
% 4. DT
% 5. SVM (rbf)





Classifiers=5;


for T=1:Classifiers

        if(T== 1)
        %gda
        gda=fitcdiscr(Xtrain,Ytrain);  
        train_predict(:,T) = predict(gda,Xtrain);
        test_predict(:,T)  = predict(gda,Xtest);
        CCR(T,:) = 1 - sum(test_predict(:,T)~= test_label)/length(test_label);
        end
    
        if(T == 2)
        %knn
        knn=fitcknn(Xtrain,Ytrain,'NumNeighbors',30);
        train_predict(:,T) = predict(knn, Xtrain);
        test_predict(:,T)  = predict(knn, Xtest);
        CCR(T,:) = 1 - sum(test_predict(:,T)~= test_label)/length(test_label);
        end

        if(T ==3)
        %NB
        nb=fitcnb(Xtrain,Ytrain);
        train_predict(:,T) = predict(nb, Xtrain);
        test_predict(:,T)  = predict(nb, Xtest);
        CCR(T,:) = 1 - sum(test_predict(:,T)~= test_label)/length(test_label);
        end

    
        if(T ==4)
        %Decision Tree
        Tree=fitctree(Xtrain,Ytrain);
        train_predict(:,T) = predict(Tree,Xtrain);
        test_predict(:,T)  = predict(Tree, Xtest);
        CCR(T,:) = 1 - sum(test_predict(:,T)~= test_label)/length(test_label);
        end

     
        if(T ==5)
        %svm
        svm=svmtrain(Xtrain,Ytrain,...
        'autoscale','false','kernelcachelimit',500000,'kernel_function','linear', ...
        'boxconstraint',2,'kktviolationlevel',0.25);
        train_predict(:,T) = svmclassify(svm, Xtrain);
        test_predict(:,T)  = svmclassify(svm, Xtest);
        CCR(T,:) = 1 - sum(test_predict(:,T)~= test_label)/length(test_label);
        end

    %Using vote as combiner
    %final vote
    train_ada_QDA(:,T)=mode(train_predict,2);
    train_ada_CCR(T) = 1- sum(train_ada_QDA(:,T) ~= train_label) / length(train_label);
   %for test set
    test_ada_QDA(:,T)=mode(test_predict,2);
    test_ada_CCR = 1- sum(test_ada_QDA(:,T) ~= test_label) / length(test_label);
    
     
end


    

new_train_data = [train_data,train_predict(:,5)];
new_test_data = [test_data,test_predict(:,5)];
new_regression=fitctree(new_train_data,train_label);
% new_regression=fitclinear(new_train_data,train_label,'Learner','logistic');
new_train_predict = predict(new_regression, new_train_data);
CCR_stacking_train = 1 -sum(new_train_predict~= train_label)/length(new_train_predict);

new_test_predict = predict(new_regression, new_test_data);
CCR_stacking_test = 1 -sum(new_test_predict~= test_label)/length(new_test_predict);


%Stacking CCR for each iteration
figure(1)
hold on
plot(1:T, train_ada_CCR)
plot(1:T, test_ada_CCR)
legend('Train Adaboosting CCR','Test Adaboosting CCR')
xlabel('Number of Iteration')
ylabel('CCR')
title('Adaboosting CCR')
hold off


test_ada_CCR = [0.7988;0.8030;0.8062;0.8220;0.7528;0.8220;0.7638];
train_ada_CCR = [0.8154;0.8030;0.8063;0.9432;0.7628;0.9432;0.7650];
figure(2)
hold on
bar(test_ada_CCR)
axe_labels = {'GDA','knn','Naive Bayes', 'Decision Tree','SVM','Stacking(Decision Tree)','stacking (LR)'};
set(gca,'xlim',[0,T+3], 'xTick',1:T+2, 'xticklabel',axe_labels);
xtickangle(45);
legend('test CCR')
xlabel('Learning Algorithm')
ylabel('CCR')
title('CCR against different learning agorithms')
hold off


