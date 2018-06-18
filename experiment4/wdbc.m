%% load Wisconsin Cancer dataset and partition data: 4:1:1. ATTENTION: it is a binary classification problem
dataset = load('./dataset/wdbc.txt');
data = dataset(:, 1:30);
label = dataset(:, 31);
y = label;

train_x = data(1:400, :);
train_y = y(1:400);
val_x = data(401: 469, :);
val_y = y(401: 469);
test_x = data(470: 569, :);
test_y = y(470: 569);

% hh
val_y = [val_y bsxfun(@minus, 1, val_y)];
train_y = [train_y bsxfun(@minus, 1, train_y)];
test_y = [test_y bsxfun(@minus, 1, test_y)];
%% preprocess the data, using the channel mean
mean = sum(train_x, 1)/400;
train_x = bsxfun(@minus, train_x, mean);
val_x = bsxfun(@minus, val_x, mean);
test_x = bsxfun(@minus, test_x, mean);

%% train neural network
%set hyper-parameters
neuron_num = [30 128 2];
learning_rate = 0.001;
batch_size = 80;
num_Epoches = 100; 
learning_rate_decay = 0.8;
weight_decay = 5e-3;

%train the network
weight = Train_NN(train_x, train_y, val_x, val_y, neuron_num, learning_rate, batch_size, num_Epoches, learning_rate_decay, weight_decay);

%% test on test_set
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
accuracy_tst = sum(m1==m2)/100;
%loss
P = exp(x);
P_sum = sum(exp(x), 2);
for i = 1: size(x, 2)-1
    P_sum = [P_sum sum(exp(x), 2)];
end
P = P./P_sum;
loss_tst = sum(-log(sum((test_y.* P), 2)))/100;
%print the result
fprintf('result on test set: test accuracy: %f, test loss: %f\n', [accuracy_tst, loss_tst]);
