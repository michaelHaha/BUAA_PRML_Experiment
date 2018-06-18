%% load Iris dataset and partition data
dataset = load('./dataset/Iris.txt');
% partition the dataset to training set, validation set and test set with
% proportion of 3:1:1 homogeneously
data = dataset(:, 1:4);
label = dataset(:, 5);
y = zeros(150, 3);
for i = 1:150
    y(i, label(i)) = 1;
end

train_x = [data(1:30, :); data(51:80, :); data(101:130, :)];
train_y = [y(1:30, :); y(51:80, :); y(101:130, :)];
val_x = [data(31:40, :); data(81:90, :); data(131:140, :)];
val_y = [y(31:40, :); y(81:90, :); y(131:140, :)];
test_x = [data(41:50, :); data(91:100, :); data(141:150, :)];
test_y = [y(41:50, :); y(91:100, :); y(141:150, :)];

%% train neural network
%set hyper-parameters
neuron_num = [4 6 3];
learning_rate = 0.5;
batch_size = 5;
num_Epoches = 200; 
learning_rate_decay = 1;
weight_decay = 5e-3;

%train the network
weight = Train_NN(train_x, train_y, val_x, val_y, neuron_num, learning_rate, batch_size, num_Epoches, learning_rate_decay, weight_decay);

% test on test_set
x = [test_x ones(length(test_x),1)];
layer_num = size(neuron_num, 2);

for i = 1: layer_num-1
    %augment data
    x = x*weight{i};
    if i ~= layer_num-1
        x = activation_Fn(x);
    end
end
%compute loss and accuracy
%accuracy
[~, m1] = max(x, [], 2);
[~, m2] = max(test_y, [], 2);
accuracy_tst = sum(m1==m2)/30;
%loss
P = exp(x);
P_sum = sum(exp(x), 2);
for i = 1: size(x, 2)-1
    P_sum = [P_sum sum(exp(x), 2)];
end
P = P./P_sum;
loss_tst = sum(-log(sum((test_y.* P), 2)))/30;
%print the result
fprintf('result on test set: test accuracy: %f, test loss: %f\n', [accuracy_tst, loss_tst]);

