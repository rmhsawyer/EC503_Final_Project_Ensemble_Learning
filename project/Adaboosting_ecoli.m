DataPreprocessing
% get_adult
Mdl1 = fitcensemble(train_data(:,1:7),train_data(:,8),'Method','AdaBoostM2','Learners','tree')
% test on training set
label = predict(Mdl1,train_data(:,1:7));
train_comatrix_tree = confusionmat(train_data(:,8),label);
tree_CCR_ADA_train= sum(diag(train_comatrix_tree))/sum(sum(train_comatrix_tree));
tree_CV_precision = train_comatrix_tree(2,2)/(train_comatrix_tree(2,2) + train_comatrix_tree(1,2));
%recall = TP/n+ = P(h(x) = 1 | Y = 1)
tree_CV_recall = train_comatrix_tree(2,2) / (train_comatrix_tree(2,2) + train_comatrix_tree(2,1));
%recall F-score = 2PR/(P+R)
tree_CV_Fscore = 2 * tree_CV_precision * tree_CV_recall / (tree_CV_precision + tree_CV_recall);
%test on test set
label = predict(Mdl1,test_data(:,1:7));
test_comatrix_tree = confusionmat(test_data(:,8),label);
tree_CCR_ADA_test = sum(diag(test_comatrix_tree))/sum(sum(test_comatrix_tree));
tree_CV_precision_test = test_comatrix_tree(2,2)/(test_comatrix_tree(2,2) + test_comatrix_tree(1,2));
%recall = TP/n+ = P(h(x) = 1 | Y = 1)
tree_CV_recall_test = test_comatrix_tree(2,2) / (test_comatrix_tree(2,2) + test_comatrix_tree(2,1));
%recall F-score = 2PR/(P+R)
tree_CV_Fscore_test = 2 * tree_CV_precision_test * tree_CV_recall_test / (tree_CV_precision_test + tree_CV_recall_test);
        
        
% % 
% Mdl1 = fitcensemble(train_data(:,1:7),train_data(:,8),'Method','AdaBoostM2','Learners','discriminant')
% % test on training set
% label = predict(Mdl1,train_data(:,1:7));
% train_comatrix_discriminant = confusionmat(train_data(:,8),label);
% CCR_ADA_discriminant = sum(diag(train_comatrix_discriminant))/sum(sum(train_comatrix_discriminant));
% disc_CCR_ADA_train = (train_comatrix_discriminant(1,1) + train_comatrix_discriminant(2,2))/sum(sum(train_comatrix_discriminant));
% disc_CV_precision = train_comatrix_discriminant(2,2)/(train_comatrix_discriminant(2,2) + train_comatrix_discriminant(1,2));
% %recall = TP/n+ = P(h(x) = 1 | Y = 1)
% disc_CV_recall = train_comatrix_discriminant(2,2) / (train_comatrix_discriminant(2,2) + train_comatrix_discriminant(2,1));
% %recall F-score = 2PR/(P+R)
% disc_CV_Fscore = 2 * disc_CV_precision * disc_CV_recall / (disc_CV_precision + disc_CV_recall);
% % test on test set
% label = predict(Mdl1,test_data(:,1:7));
% test_comatrix_discriminant = confusionmat(test_data(:,8),label);
% CCR_ADA_discriminant_test = sum(diag(test_comatrix_discriminant))/sum(sum(test_comatrix_discriminant));
% disc_CCR_ADA_test = (test_comatrix_discriminant(1,1) + test_comatrix_discriminant(2,2))/sum(sum(test_comatrix_discriminant));
% disc_CV_precision_test = test_comatrix_discriminant(2,2)/(test_comatrix_discriminant(2,2) + test_comatrix_discriminant(1,2));
% %recall = TP/n+ = P(h(x) = 1 | Y = 1)
% disc_CV_recall_test = test_comatrix_discriminant(2,2) / (test_comatrix_discriminant(2,2) + test_comatrix_discriminant(2,1));
% %recall F-score = 2PR/(P+R)
% disc_CV_Fscore_test = 2 * disc_CV_precision_test * disc_CV_recall_test / (disc_CV_precision_test + disc_CV_recall_test);
%     
% 
%You cannot use KNN for ensemble learning other than random subspace.
% Mdl1 = fitcensemble(train_data,train_label,'Method','AdaBoostM1','Learners','knn', 'CategoricalPredictors','all')
% label = predict(Mdl1,train_data);
% train_comatrix_knn = confusionmat(train_label,label);