function error = err_m(w, predict, label)
    % initial 
    N = length(predict);
    tmp1 = 0;
    tmp2 = 0;
    for  i = 1 : N
        tmp1 = tmp1 + w(i) * (predict ~= label);
        tmp2 = tmp2 + w(i);
    end
    error = tmp1/tmp2;
end