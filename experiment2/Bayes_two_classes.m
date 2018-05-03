%data input
X1 = [-3.9847 -3.5549 -1.2401 -0.9780 -0.7932 -2.8531 -2.7605 -3.7287 -3.5414 -2.2692 -3.4549 -3.0752 -3.9934  -0.9780 -1.5799 -1.4885 -0.7431 -0.4221 -1.1186 -2.3462 -1.0826 -3.4196 -1.3193 -0.8367 -0.6579 -2.9683];
X2 = [2.8792 0.7932 1.1882 3.0682 4.2532 0.3271 0.9846 2.7648 2.6588];
P_w1 = 0.9;
P_w2 = 0.1;
Lambda_a1_w2 = 1;
Lambda_a2_w1 = 6;

x = [-8:0.01:8];
%compute and plot conditional probabilty density
std1 = std(X1);
mean1 = mean(X1);
std2 = std(X2);
mean2 = mean(X2);
p_X_w1 = (1/(sqrt(2*pi)*std1))*exp(-1*(x-mean1).^2/(2*std1^2));
p_X_w2 = (1/(sqrt(2*pi)*std2))*exp(-1*(x-mean2).^2/(2*std2^2));
figure(1);
plot(x,p_X_w1);hold on;
plot(x,p_X_w2);

% compute and plot posterior probablity 
p_X_w1_joint = p_X_w1*P_w1;
p_X_w2_joint = p_X_w2*P_w2;
p_w1_X = p_X_w1_joint./(p_X_w1_joint+p_X_w2_joint);
p_w2_X = p_X_w2_joint./(p_X_w1_joint+p_X_w2_joint);
figure(2);
plot(x,p_w1_X);hold on;
plot(x,p_w2_X);

%Min error Bayes decision boundary
for i = 1:length(x)
    if(p_w1_X(i)<p_w2_X(i))
        disp(['the min error decision boundary is:',num2str(x(i))]);
        plot(x(i),p_w1_X(i),'ob', 'LineWidth', 2);
        break;
    end
end

%Min risk Bayes decision boundary
Risk_a1 = p_w2_X*Lambda_a1_w2;
Risk_a2 = p_w1_X*Lambda_a2_w1;
figure(3);
plot(x,Risk_a1);hold on;
plot(x,Risk_a2);
for i = 1:length(x)
    if(Risk_a1(i)>Risk_a2(i))
        disp(['the min risk decision boundary is:',num2str(x(i))]);
        plot(x(i),Risk_a1(i),'ob', 'LineWidth', 2);
        break;
    end
end
