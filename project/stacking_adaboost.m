
% get_adult();
load train_data.mat
load train_label.mat
load test_data.mat
load test_label.mat

% Make data label -1 and 1
n= length(train_label);
for i = 1:n
    if(train_label(i)== 0)
        train_label(i) = -1;
    end
end

% Make data label -1 and 1
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
% 4. Logistic Regression
% 5. SVM (rbf)
weak_learner=fitcdiscr(train_data,train_label);
predicted=predict(weak_learner, train_data);
CCR3_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR3 = 1- sum(predicted ~= test_label)/length(predicted);

weak_learner=fitcknn(train_data,train_label,'NumNeighbors',30);
predicted=predict(weak_learner, train_data);
CCR4_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR4 = 1- sum(predicted ~= test_label)/length(predicted);

weak_learner=fitcnb(train_data,train_label);
predicted=predict(weak_learner, train_data);
CCR5_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR5 = 1- sum(predicted ~= test_label)/length(predicted);

weak_learner=fitclinear(train_data,train_label,'Learner','logistic');
predicted=predict(weak_learner, train_data);
CCR6_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR6 = 1- sum(predicted ~= test_label)/length(predicted);

weak_learner=fitcsvm(train_data,train_label,'KernelFunction','rbf');
predicted=predict(weak_learner, train_data);
CCR7_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR7 = 1- sum(predicted ~= test_label)/length(predicted);

% Choosen Weak classifiers as stacking method:
% 1. GDA
% 2. knn (NumNeighbors = 30)
% 3. Naive Bayes
% 4. Logistic Regression
% 5. SVM (rbf)

Xtrain=train_data;
Ytrain =train_label;
Xtest = test_data;
Ytest = test_label;

N=size(Xtrain,1);
a=[Xtrain Ytrain];
% Initiate weights
W=(1/N)*ones(N,1);
Dt=[]; h_=[];

Classifiers=5;
eps=zeros(Classifiers,1);



for T=1:Classifiers
    w_min=min(W);
    w_max=max(W);
    
    for i=1:length(W)
        p = (w_max-w_min)*rand(1) + w_min;
        
        if W(i)>=p
            d(i,:)=a(i,:);
        end
        
        t=randi(size(d,1));
        Dt=[Dt ;d(t,:)];
    end

    X=Dt(:,1:end-1);
    Y=Dt(:,end);

        if(T== 1)
        %gda
        gda=fitcdiscr(X,Y);
        gda_out=predict(gda, X);
        h=gda_out;
        Dt=Dt(length(Dt)+1:end,:);
        
        train_predict(:,T) = predict(gda, train_data);
        test_predict(:,T)  = predict(gda, test_data);
        end
    
        if(T == 2)
        %knn
        knn=fitcknn(X,Y,'NumNeighbors',30);
        knn_out=predict(knn, X);
        h=knn_out;
        Dt=Dt(length(Dt)+1:end,:);
        train_predict(:,T) = predict(knn, train_data);
        test_predict(:,T)  = predict(knn, test_data);
        end

        if(T ==3)
        %NB
        nb=fitcnb(X,Y);
        nb_out=predict(nb, X);
        h=nb_out;
        Dt=Dt(length(Dt)+1:end,:);
        train_predict(:,T) = predict(nb, train_data);
        test_predict(:,T)  = predict(nb, test_data);
        end

    
        if(T ==4)
        %logistic regression
        regression=fitclinear(X,Y,'Learner','logistic');
        linear_out=predict(regression, X);
        h=linear_out;
        Dt=Dt(length(Dt)+1:end,:);
        train_predict(:,T) = predict(regression, train_data);
        test_predict(:,T)  = predict(regression, test_data);
        end

     
        if(T ==5)
        %svm
        svm=fitcsvm(X,Y,'KernelFunction','rbf');
        svm_out=predict(svm, X);
        h=svm_out;
        Dt=Dt(length(Dt)+1:end,:);
        train_predict(:,T) = predict(svm, train_data);
        test_predict(:,T)  = predict(svm, test_data);
        end

    
    h_=[h_ h];

    % weighted error
    for i=1:length(Y)
        if (h_(i,T)~=Y(i))
            eps(T)=eps(T)+W(i,:); 
        end  
    end
    
    % Hypothesis weight
    alpha(T)=0.5*log((1-eps(T))/eps(T));
    
    % Update weights
    W=W.*exp((-1).*Y.*alpha(T).*h);
    W=W./sum(W);
    
    % final vote
    train_ada_QDA(:,T)=sign(train_predict*alpha');
    train_ada_CCR(T) = 1- sum(train_ada_QDA(:,T) ~= train_label) / length(train_label);
    % for test set
    test_ada_QDA(:,T)=sign(test_predict*alpha');
    test_ada_CCR(T) = 1- sum(test_ada_QDA(:,T) ~= test_label) / length(test_label);
        
end



new_train_data = train_predict(:,5);
new_test_data = test_predict(:,5);
new_regression=fitclinear(new_train_data,train_label,'Learner','logistic');
new_train_predict = predict(new_regression, new_train_data);
CCR_stacking_train = 1 -sum(new_train_predict~= train_label)/length(new_train_predict);

new_test_predict = predict(new_regression, new_test_data);
CCR_stacking_test = 1 -sum(new_test_predict~= test_label)/length(new_test_predict);





