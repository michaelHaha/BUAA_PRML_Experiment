learning_rate = 0.3;
iter_res = zeros(1,10);
num_sep = 1;
for sep = 0.5:0.5:5  
    iter_tmp = 0;
    for num = 1:100   %the result should be calculated on average on 100 experiments
        iterations = Experiment_1(sep, learning_rate,2);
        iter_tmp = iter_tmp+iterations;
    end
    iter_res(num_sep) = iter_tmp/100;
    num_sep = num_sep+1;
end