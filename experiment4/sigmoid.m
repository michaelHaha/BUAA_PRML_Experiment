%% sigmoid for sigmoid loss
function y = sigmoid(x)
    y = zeros(size(x));
    for i = 1:length(x)
        y(i) = 1/(1+exp(-x(i)));
    end
end