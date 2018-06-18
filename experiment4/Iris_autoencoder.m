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
test_x = [data(31:50, :); data(81:100, :); data(131:150, :)];

%% train neural network
%set hyper-parameters
neuron_num = [4 3 4];
learning_rate = 0.001;
batch_size = 5;
num_Epoches = 200; 
learning_rate_decay = 0.9;

%train the network
weight = Train_AutoEncoder(train_x, neuron_num, learning_rate, batch_size, num_Epoches, learning_rate_decay);

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
% compute loss and get ready for BP
x_init = test_x;
loss = 0.5 * (sum(sum((x - x_init).^2)) / 60);

%print the result
fprintf('result on test set: test loss: %f\n', loss);


%%
x1 = round(x*100)/100;
test_x1 = round(test_x*100)/100;
x1 = vpa(x, 3)
x1