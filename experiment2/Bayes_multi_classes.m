%% generate dataset: 2 dimensional 3 classes
mean1_construct = [-2 -3];cov1_construct = [1 0.5; 0.5 2];
mean2_construct = [2 3];  cov2_construct = [2 0.75; 0.75 1];
mean3_construct = [3 -3]; cov3_construct = [1 0.25; 0.25 1];
X1 = mvnrnd(mean1_construct, cov1_construct, 100); %generate the dataset by multivariate normal random numbers
X2 = mvnrnd(mean2_construct, cov2_construct, 100);
X3 = mvnrnd(mean3_construct, cov3_construct, 100);
figure(1);
plot(X1(:, 1), X1(:, 2), '.g');hold on;
plot(X2(:, 1), X2(:, 2), 'ob');hold on;
plot(X3(:, 1), X3(:, 2), '*r');

%% assign the prior probablity
P_w1 = 0.7;
P_w2 = 0.2;
P_w3 = 0.1;

[x, y] = meshgrid(-8 : 0.1 : 8);
%% compute and plot conditional probablity
mean1 = mean(X1); cov1 = cov(X1);
mean2 = mean(X2); cov2 = cov(X2);
mean3 = mean(X3); cov3 = cov(X3);
P_X_w1 = reshape(mvnpdf([x(:), y(:)], mean1, cov1),size(x));
P_X_w2 = reshape(mvnpdf([x(:), y(:)], mean2, cov2),size(x));
P_X_w3 = reshape(mvnpdf([x(:), y(:)], mean3, cov3),size(x));
figure(2);
surf(x,y,P_X_w1);hold on;
surf(x,y,P_X_w2);hold on;
surf(x,y,P_X_w3);

%% compute posterior probablity and make min error decision boundary
p_X_w1_joint = P_X_w1*P_w1;
p_X_w2_joint = P_X_w2*P_w2;
p_X_w3_joint = P_X_w3*P_w3;
p_w1_X = p_X_w1_joint./(p_X_w1_joint+p_X_w2_joint+p_X_w3_joint);
p_w2_X = p_X_w2_joint./(p_X_w1_joint+p_X_w2_joint+p_X_w3_joint);
p_w3_X = p_X_w3_joint./(p_X_w1_joint+p_X_w2_joint+p_X_w3_joint);
figure(3);
surf(x,y,p_w1_X);hold on;
surf(x,y,p_w2_X);hold on;
surf(x,y,p_w3_X);

Region_result = zeros(size(x));
for i = 1:size(x,1)
    for j = 1:size(x,1)
        [~,Class] = min([p_w1_X(i,j) p_w2_X(i,j) p_w3_X(i,j)]);
        Region_result(i,j) = Class;
    end
end
figure(4);
surf(x,y,Region_result);

%% compute posterior probablity and make min error decision boundary
% define risk matrix
Lambda = [0 2 1; 2 0 3; 3 5 0];
Risk1 = Lambda(1, 1) * p_w1_X + Lambda(1, 2) * p_w2_X + Lambda(1, 3) * p_w3_X;
Risk2 = Lambda(2, 1) * p_w1_X + Lambda(2, 2) * p_w2_X + Lambda(2, 3) * p_w3_X;
Risk3 = Lambda(3, 1) * p_w1_X + Lambda(3, 2) * p_w2_X + Lambda(3, 3) * p_w3_X;
figure(5);
surf(x,y,Risk1);hold on;
surf(x,y,Risk2);hold on;
surf(x,y,Risk3);

Region_result = zeros(size(x));
for i = 1:size(x,1)
    for j = 1:size(x,1)
        [~,Class] = min([Risk1(i,j) Risk2(i,j) Risk3(i,j)]);
        Region_result(i,j) = Class;
    end
end
figure(6);
surf(x,y,Region_result);