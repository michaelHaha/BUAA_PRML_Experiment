iter_res = zeros(10,10);
num_sep = 1;
for sep = 0.5:0.5:5  
    num_lr = 1;
    for learning_rate = 0.1:0.1:1
        iter_tmp = 0;
        for num = 1:100   %the result should be calculated on average on 100 experiments
            iterations = Experiment_1(sep, learning_rate,26);
            iter_tmp = iter_tmp+iterations;
        end
        iter_res(num_sep,num_lr) = roundn(iter_tmp/100,-3);
        num_lr = num_lr+1;
    end
    num_sep = num_sep+1;
end