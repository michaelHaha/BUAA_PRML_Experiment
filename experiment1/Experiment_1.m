function [iterations] = Experiment_1(sep, learning_rate, dimension)
%% generate dataset
num_samples = 200;
%define seperation line: x1+x2=sep/x1+x2=-sep
w1 = [];
w2 = [];

%generate dataset and assign their identity according to their position
num_gen_w1 = 0;
num_gen_w2 = 0;

%construct balanced dataset
while(1)
    data = rand(1,dimension)*40-20; 
    if (sum(data) > sep)
        w1 = [w1; data];
        num_gen_w1 = num_gen_w1+1;
    end
    if((num_gen_w1 == num_samples/2))
        break;
    end
end
while(1)
    data = rand(1,dimension)*40-20; 
    if (sum(data) < -sep)
        w2 = [w2; data];
        num_gen_w2 = num_gen_w2+1;
    end
    if((num_gen_w2 == num_samples/2))
        break;
    end
end
%         plot(w1(:, 1), w1(:, 2), 'or', 'LineWidth', 1.5);hold on;
%         plot(w2(:, 1), w2(:, 2), '+b', 'LineWidth', 1.5);hold on;
%% train the perceptron
[W, iterations, is_Converge] = Train_perceptron(w1,w2,num_samples,learning_rate);
if (is_Converge == true)
    if(dimension == 2)
        %display 2d
        plot(w1(:, 1), w1(:, 2), 'or', 'LineWidth', 1.5);hold on;
        plot(w2(:, 1), w2(:, 2), '+b', 'LineWidth', 1.5);hold on;
        X = [-25:0.05:25];
        Y= -1*(W(1)/W(2))*X-(W(3)/W(2));
        plot(X,Y);
        
    elseif(dimension == 3)
        %display 3-d 
        plot3(w1(:, 1), w1(:, 2),w1(:, 3), 'or', 'LineWidth', 1.5);hold on;
        plot3(w2(:, 1), w2(:, 2),w2(:, 3), '+b', 'LineWidth', 1.5)
        X1 = linspace(-20,20,400);
        X2 = linspace(-20,20,400);
        [X1,X2]=meshgrid(X1,X2);
        Y= -1*(W(1)/W(3))*X1-(W(2)/W(3))*X2-(W(4)/W(3));
        surf(X1,X2,Y);
    end
disp(['W is :',num2str(W')]);  
end

end
    

