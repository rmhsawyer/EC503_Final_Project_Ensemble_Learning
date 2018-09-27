get_adult();

n= length(train_label);
for i = 1:n
    if(train_label(i)== 0)
        train_label(i) = -1;
    end
end
        
randomIndex = randperm(n);
train_index = randomIndex(1:0.7*n);
test_index = randomIndex(0.7*n:n);
train_data1 =train_data(train_index,:);
train_label1 = train_label(train_index);
originial_data = train_data;
originial_label = train_label;
w = ones (n,1)/n;

for i= 1:5
    if(i~=1)

        [w, index] = sort(w,'descend');
        train_data =train_data(index,:);
        train_label =train_label(index,:);

        train_data1 = train_data(1:22792,:);
        train_label1 = train_label(1:22792,:);
    end
    Mdl = fitcdiscr(train_data1,train_label1,'DiscrimType','linear');
    Y_predict = predict (Mdl,train_data);
    error = sum(w(Y_predict~=train_label)) /sum (w) ;
    a(i) = log( (1-error)/error);
    w = w.* exp((train_label ~= Y_predict) * a(i));
    w =w./ sum(w);
    
    CCR(i) = 1 - sum(Y_predict~=train_label)/length(Y_predict);
    original_predict = predict (Mdl,originial_data);
    local_Y(:,i) = original_predict;
    
end

final_predict = zeros(n,1);
for i = 1:5
    final_predict = final_predict+ a(i) .* local_Y(:,i);
end

final_predict = sign(final_predict);
final_CCR = 1 - sum(final_predict~=originial_label)/length(final_predict);

