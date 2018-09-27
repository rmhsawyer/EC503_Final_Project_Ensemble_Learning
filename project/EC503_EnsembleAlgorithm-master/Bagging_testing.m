%% Testing 
clear all; clc;
warning off; 

%Binary Class // Adult Data
get_adult
test_set_predicted_Y = Ensemble_bagging(train_data, train_label, test_data, 1, 2);
conf = confusionmat(test_label, test_set_predicted_Y);
testCRR_binaryclass = sum(diag(conf)) / sum(sum(conf))

clear train_data test_data

%Muticlass // Ecoli Data
DataPreprocessing
test_set_predicted_Y = Ensemble_bagging(train_data(:,1:end-1), train_data(:,end), test_data(:,1:end-1), 1, 8);
conf = confusionmat(test_data(:,end), test_set_predicted_Y);
testCRR_muticlass = sum(diag(conf)) / sum(sum(conf))