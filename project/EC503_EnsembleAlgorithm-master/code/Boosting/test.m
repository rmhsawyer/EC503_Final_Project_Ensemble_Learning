get_adult;
Mdl1 = fitcensemble(train_data,train_label,'Method','AdaBoostM1','Learners','tree')
label = predict(Mdl1,train_data);
train_comatrix_tree = confusionmat(train_label,label);
tree_CCR_ADA = (train_comatrix_tree(1,1) + train_comatrix_tree(2,2))/sum(sum(train_comatrix_tree));
tree_CV_precision = train_comatrix_tree(2,2)/(train_comatrix_tree(2,2) + train_comatrix_tree(1,2));
%recall = TP/n+ = P(h(x) = 1 | Y = 1)
tree_CV_recall = train_comatrix_tree(2,2) / (train_comatrix_tree(2,2) + train_comatrix_tree(2,1));
%recall F-score = 2PR/(P+R)
tree_CV_Fscore = 2 * tree_CV_precision * tree_CV_recall / (tree_CV_precision + tree_CV_recall);
        

Mdl1 = fitcensemble(train_data,train_label,'Method','AdaBoostM1','Learners','discriminant')
label = predict(Mdl1,train_data);
train_comatrix_discriminant = confusionmat(train_label,label);
CCR_ADA_discriminant = (train_comatrix_discriminant(1,1) + train_comatrix_discriminant(2,2))/sum(sum(train_comatrix_discriminant));
disc_CCR_ADA = (train_comatrix_discriminant(1,1) + train_comatrix_discriminant(2,2))/sum(sum(train_comatrix_discriminant));
disc_CV_precision = train_comatrix_discriminant(2,2)/(train_comatrix_discriminant(2,2) + train_comatrix_discriminant(1,2));
%recall = TP/n+ = P(h(x) = 1 | Y = 1)
disc_CV_recall = train_comatrix_discriminant(2,2) / (train_comatrix_discriminant(2,2) + train_comatrix_discriminant(2,1));
%recall F-score = 2PR/(P+R)
disc_CV_Fscore = 2 * disc_CV_precision * disc_CV_recall / (disc_CV_precision + disc_CV_recall);
    

%You cannot use KNN for ensemble learning other than random subspace.
% Mdl1 = fitcensemble(train_data,train_label,'Method','AdaBoostM1','Learners','knn', 'CategoricalPredictors','all')
% label = predict(Mdl1,train_data);
% train_comatrix_knn = confusionmat(train_label,label);