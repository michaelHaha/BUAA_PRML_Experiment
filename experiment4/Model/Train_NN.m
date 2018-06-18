function weight = Train_NN(train_x, train_y, val_x, val_y, neuron_num, learning_rate, batch_size, num_Epoches, learning_rate_decay, weight_decay)
%% 
%Inputs:
%training_set, labels: train_x, train_label
%hyperparameters: learning_rate, batch_size, number of epoches to
%train, learning_rate_decay, weight_decay

%Outputs; 
%Weights; both weight matrice and biases

%% preparation for training
% Ensure neuron number match input data
assert(size(train_x, 2) == neuron_num(1));
assert(size(train_y, 2) == neuron_num(end));

%get number of inputs and layers
Num_Inputs_trn = length(train_x);
Num_Inputs_val = length(val_x);
layer_num = length(neuron_num);

%compute iterations per epoch
iters_per_epoch = (Num_Inputs_trn/batch_size);

%outputs_before_activation of every layer, used for BP
outputs_before_activation = cell(1,layer_num);  

%augment training set
train_x_aug = [train_x ones(Num_Inputs_trn, 1)];

%initiallize Augmented weights with Gaussian distribution with mean 0, deviation 0.01
% Notice: weight{i} corresponds to the (i+1) layer
weight = cell(1, layer_num - 1);
for i = 1:layer_num-1
    if i ~= layer_num-1
        weight{i} = randn(neuron_num(i)+1, neuron_num(i+1)+1)/100;
        weight{i}(neuron_num(i)+1, :) = 0;
    else
        weight{i} = randn(neuron_num(i)+1, neuron_num(i+1))/100;
        weight{i}(neuron_num(i)+1, :) = 0;
    end
end
% for i = 1:layer_num-1
%     if i ~= layer_num-1
%         weight{i} = zeros(neuron_num(i)+1, neuron_num(i+1)+1)/10; 
%     else
%         weight{i} = zeros(neuron_num(i)+1, neuron_num(i+1))/10;
%     end
% end

% D_weight: restore the gradient for each weight matrix in weight(layer: 1 - layer_num-1)
D_weight = cell(1,layer_num-1);

% D_output:restore the gradient for output in each layer(layer: 2 - layer_num-1)
D_output = cell(1,layer_num-1);
% for i = 1: layer_num-1
%     D_weight{i} = zeros(neuron_num(i)+1, neuron(i+1));
% end

% record:
loss = 0.0;
accuracy = 0.0;
global_step = 0;

trn_acc_rec = zeros(num_Epoches,1);
trn_los_rec = zeros(num_Epoches,1);
val_acc_rec = zeros(num_Epoches,1);
val_los_rec = zeros(num_Epoches,1);

%% start training
fprintf('Training...\n');

for epoch = 1 : num_Epoches
    %shuffle data every epoch
    shuffle = randperm(size(train_x_aug, 1));
    train_x_aug_shuf = train_x_aug(shuffle,:);
    train_y_shuf = train_y(shuffle, :);
    fprintf('strating training %d epoch\n', epoch);
    
    acc_per_epoch = 0;
    los_per_epoch = 0;

    for iter = 1:iters_per_epoch
        % get batch and its labels
        batch = train_x_aug_shuf(((batch_size*(iter-1))+1):(batch_size*(iter)), :);
        y = train_y_shuf(((batch_size*(iter-1))+1):(batch_size*(iter)), :);
        
        %feed forward and record inputs(outputs) of each stage
        outputs_before_activation{1} = batch;
        for i = 2: layer_num
            %first feed the outputs of last layer to activation function
            %and then feed it to this layer
            if i == 2
                outputs_before_activation{i} = (outputs_before_activation{i-1})*weight{i-1};            
            else
                outputs_after_activation = activation_Fn(outputs_before_activation{i-1});
%                 outputs_after_activation = [outputs_after_activation ones(length(outputs_after_activation), 1)];
                outputs_before_activation{i} = outputs_after_activation * weight{i-1}; 
            end
        end
        
        % output and compute Loss
        output = outputs_before_activation{layer_num};
        % two kinds of loss: sigmoid loss(one-class classification) and
        % cross-entropy loss(multi-class classification)
        
        if size(output, 2) == 1   
            %accuracy
            accuracy = (sum((output > 0) == y))/batch_size;

            %sigmoid loss
            loss = sum((log(sigmoid(output))).*y)+sum((ones(size(y))-y).*(log(ones(size(output))-sigmoid(output))));
            loss = -loss/batch_size;

            % preparing update for the weights
            delta_output = (output-y)/batch_size;
            D_output{end} = delta_output;
        else
            %accuracy
            [~,max_position] = max(output, [], 2);
            [~, max_position_y] = max(y, [], 2);
            accuracy = sum(max_position == max_position_y)/batch_size;
            
            %cross-entropy loss
            P = exp(output);
            P_sum = sum(exp(output), 2);
            for i = 1: size(output, 2)-1
                P_sum = [P_sum sum(exp(output), 2)];
            end
            P = P./P_sum;
            loss = sum(-log(sum((y.* P), 2)))/batch_size;
            
            %preparing update for the weights
            P_cut = y;
            delta_output = (P-P_cut)/batch_size;
            D_output{end} = delta_output;
        end
        % need to add L-2 loss to the total loss
        for i = 1: layer_num-1
            dis = weight{i}.^2;
            loss = loss+sum(dis(:))*0.5*weight_decay;
        end
        
        %back propagation: compute the gradient for each weight matrix
        for i = layer_num-1:-1:1
            if i ~= 1
                outputs_after_activation = activation_Fn(outputs_before_activation{i});
%                 outputs_after_activation = [outputs_after_activation ones(length(outputs_after_activation), 1)];
            else
                outputs_after_activation = outputs_before_activation{i};
            end
            D_weight{i} = outputs_after_activation' * D_output{i};
            D_weight{i} = D_weight{i} + weight_decay * weight{i};
            if i ~= 1
                D_output{i-1} = D_output{i} * (weight{i}');
                %let the gradient pass through the activation function:
                %relu function, partial derivative is 1 for positive inputs
                %and 0 for negative outputs
                D_output{i-1} = (outputs_after_activation>0) .* D_output{i-1};
            end
        end
        
        %update weights
        for i = 1:layer_num-1
            weight{i} = weight{i} - learning_rate * D_weight{i};
        end
        
        %record total iterations
        global_step = global_step+1;
        
        %report result every 100 iterations
        if mod(global_step, 2) == 0
            fprintf('step: %d, learning rate: %f, accuracy: %f, loss: %f\n', [global_step, learning_rate, accuracy, loss]);            
        end
        
        %record accuracy and loss for this iteration
        acc_per_epoch = acc_per_epoch + accuracy;
        los_per_epoch = los_per_epoch + loss;
    end
    
    % AFTER ONE EPOCH 
    % learning rate decay: learning rate decay every 2 epoches
    if mod(epoch, 20) == 0
        learning_rate = learning_rate*learning_rate_decay;
    end
    
    %test on validation set for evaluation every epoch
    x = [val_x ones(length(val_x),1)];
    for i = 1: layer_num-1
        %augment data
        x = x*weight{i};
        if i ~= layer_num-1
            x = activation_Fn(x);
        end
    end
    
    %compute loss and accuracy
    if size(x, 2) == 1
        %accuracy
        accuracy_val = (sum((x > 0) == val_y))/Num_Inputs_val;
        %loss
        loss_val = sum((log(sigmoid(x))).*val_y)+sum((ones(size(val_y))-val_y).*(log(ones(size(x))-sigmoid(x))));
        loss_val = -loss_val/Num_Inputs_val;
    else
        %accuracy
        [~, m1] = max(x, [], 2);
        [~, m2] = max(val_y, [], 2);
        accuracy_val = sum(m1==m2)/Num_Inputs_val;
        %cross-entropy loss
        P = exp(x);
        P_sum = sum(exp(x), 2);
        for i = 1: size(x, 2)-1
            P_sum = [P_sum sum(exp(x), 2)];
        end
        P = P./P_sum;
        loss_val = sum(-log(sum((val_y.* P), 2)))/Num_Inputs_val;
    end
    %print the result
    fprintf('result of %d epoches: val accuracy: %f, val loss: %f\n', [epoch, accuracy_val, loss_val]);
    
    %record the trn accuracy/loss and validation accruacy/loss
    trn_acc_rec(epoch) = acc_per_epoch/iters_per_epoch;
    trn_los_rec(epoch) = los_per_epoch/iters_per_epoch;
    val_acc_rec(epoch) = accuracy_val;
    val_los_rec(epoch) = loss_val;
end

%% When finish training, plot train and validation record
%plot(trn_los_rec); hold on; title('training process with different learning rate'); legend(['lr:' num2str(learning_rate)]);
% figure(1);
% plot(trn_acc_rec); hold on; plot(trn_los_rec); title('Train accuracy and loss history'); legend('Accuracy', 'Loss');xlabel('Epoches');
% figure(2);
% plot(val_acc_rec); hold on; plot(val_los_rec); title('Validation accuracy and loss history'); legend('Accuracy', 'Loss');xlabel('Epoches');
end




