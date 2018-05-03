function [W, num_epoches, is_Converge] = Train_perceptron(w1, w2, num_samples, learning_rate)
iterations_per_epoch = num_samples;
W = rand((size(w1,2)+1),1);   
is_Converge = false;
max_iterations = 100000;

iterations = 0;
num_epoches = 0;
while(is_Converge == false && iterations < max_iterations)
    is_Converge = true;
    %shuffle the data before every epoch
    data_list = randperm(num_samples);
    for step = 1:iterations_per_epoch
        % dequeue data
        data_index = data_list(step);
        if (data_index <= num_samples/2)    %data from w1
            data = [w1(data_index,:) 1];         %standard augmented data
        else
            data = -[w2(data_index-num_samples/2,:) 1];
        end
        
        %inference
        y = data*W;
        
        %update weights
        if (y <= 0)
            W = W+learning_rate*data';
            is_Converge = false;
        end
        iterations = iterations+1;
        
        if (iterations >= max_iterations)
            disp('the dataset is not linear separable!');
            break;
        end
    end
    num_epoches = num_epoches+1;
    if (is_Converge == true)
        disp(['the perceptron is converged,iterations:',num2str(num_epoches)]);
    end
end
end