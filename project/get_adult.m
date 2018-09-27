load census1994
train_adult_string = string(table2cell(adultdata));
% missing data
train_adult_string = fillmissing(train_adult_string,'previous');
% missing_data = ismissing(train_adult_string);
% missing_data = sum(missing_data,2);
% missing_data = (missing_data~=0);
% train_adult_string(missing_data,:) = [];
[N,D] = size(train_adult_string);
% Feature that are not numerical 

workclass = unique(train_adult_string(:,2));
% workc = categories(unique(table2array(adultdata(:,2))));
for i = 1:length(workclass)
    class = string(workclass(i));
    str_i = num2str(i);
    train_adult_string(:,2) = strrep(train_adult_string(:,2),class,str_i);
end
education =  unique(train_adult_string(:,4));
for i = 1:length(education)
    class = string(education(i));
    str_i = num2str(i);
    train_adult_string(:,4) = strrep(train_adult_string(:,4),class,str_i);
end
marital_status =  unique(train_adult_string(:,6));
for i = 1:length(marital_status)
    class = string(marital_status(i));
    str_i = num2str(i);
    train_adult_string(:,6) = strrep(train_adult_string(:,6),class,str_i);
end
occupation =  unique(train_adult_string(:,7));
for i = 1:length(occupation)
    class = string(occupation(i));
    str_i = num2str(i);
    train_adult_string(:,7) = strrep(train_adult_string(:,7),class,str_i);
end
relationship =  unique(train_adult_string(:,8));
for i = 1:length(relationship)
    class = string(relationship(i));
    str_i = num2str(i);
    train_adult_string(:,8) = strrep(train_adult_string(:,8),class,str_i);
end
race =  unique(train_adult_string(:,9));
for i = 1:length(race)
    class = string(race(i));
    str_i = num2str(i);
    train_adult_string(:,9) = strrep(train_adult_string(:,9),class,str_i);
end
sex =  unique(train_adult_string(:,10));
for i = 0:length(sex)-1
    class = string(sex(i+1));
    str_i = num2str(i+1);
    train_adult_string(:,10) = strrep(train_adult_string(:,10),class,str_i);
end
native_country =  unique(train_adult_string(:,14));
for i = 1:length(native_country)
    class = string(native_country(i));
    str_i = num2str(i);
    train_adult_string(:,14) = strrep(train_adult_string(:,14),class,str_i);
end
train_data = str2double(train_adult_string(:,1:14));
train_class =  unique(train_adult_string(:,15));
for i = 0:length(train_class)-1
    class = string(train_class(i+1));
    str_i = num2str(i);
    train_adult_string(:,15) = strrep(train_adult_string(:,15),class,str_i);
end
train_label = str2double(train_adult_string(:,15));
clear adultdata class D Description i missing_data N str_i train_adult_string

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_adult_string = string(table2cell(adulttest));
% missing data
test_adult_string = fillmissing(test_adult_string,'previous');
% missing_data = ismissing(train_adult_string);
% missing_data = sum(missing_data,2);
% missing_data = (missing_data~=0);
% train_adult_string(missing_data,:) = [];
[N,D] = size(test_adult_string);
% Feature that are not numerical 

% workc = categories(unique(table2array(adultdata(:,2))));
for i = 1:length(workclass)
    class = string(workclass(i));
    str_i = num2str(i);
    test_adult_string(:,2) = strrep(test_adult_string(:,2),class,str_i);
end

for i = 1:length(education)
    class = string(education(i));
    str_i = num2str(i);
    test_adult_string(:,4) = strrep(test_adult_string(:,4),class,str_i);
end

for i = 1:length(marital_status)
    class = string(marital_status(i));
    str_i = num2str(i);
    test_adult_string(:,6) = strrep(test_adult_string(:,6),class,str_i);
end

for i = 1:length(occupation)
    class = string(occupation(i));
    str_i = num2str(i);
    test_adult_string(:,7) = strrep(test_adult_string(:,7),class,str_i);
end

for i = 1:length(relationship)
    class = string(relationship(i));
    str_i = num2str(i);
    test_adult_string(:,8) = strrep(test_adult_string(:,8),class,str_i);
end

for i = 1:length(race)
    class = string(race(i));
    str_i = num2str(i);
    test_adult_string(:,9) = strrep(test_adult_string(:,9),class,str_i);
end

for i = 0:length(sex)-1
    class = string(sex(i+1));
    str_i = num2str(i);
    test_adult_string(:,10) = strrep(test_adult_string(:,10),class,str_i);
end

for i = 1:length(native_country)
    class = string(native_country(i));
    str_i = num2str(i);
    test_adult_string(:,14) = strrep(test_adult_string(:,14),class,str_i);
end
test_data = str2double(test_adult_string(:,1:14));
test_class =  unique(test_adult_string(:,15));
for i = 0:length(test_class)-1
    class = string(test_class(i+1));
    str_i = num2str(i);
    test_adult_string(:,15) = strrep(test_adult_string(:,15),class,str_i);
end
test_label = str2double(test_adult_string(:,15));
clear adulttest class D Description i missing_data N str_i test_adult_string
clear train_class test_class
