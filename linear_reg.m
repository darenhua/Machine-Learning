function h = computeHypothesis(xcoord)
%COMPUTEHYPOTHESIS Compute Hypothesis for linear regression
    h = theta(2)*xcoord + theta(1);
end

function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
m = length(y); % number of training examples
J = 0;
for i = 1:m
    xcoord = X(i,2);
    ycoord = y(i,1);
    h = computeHypothesis(xcoord);
    squareError = (h - ycoord)^2;
    J = J + squareError;    
end    
J = (J/(2*m));
end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
summation0 = 0;
summation1 = 0;
    for j = 1:m
        xcoord = X(j,2);
        ycoord = y(j,1);
        h = computeHypothesis(xcoord);
        sum = h - ycoord;     
        derive0 = sum*X(j,1);
        derive1 = sum*X(j,2);
        summation0 = summation0 + derive0;
        summation1 = summation1 + derive1;
    end
        temp0 = theta(1) - alpha*(1/m)*summation0;
        temp1 = theta(2) - alpha*(1/m)*summation1;
        theta(1) = temp0;
        theta(2) = temp1;  
    J_history(iter) = computeCost(X, y, theta);

end

end

data = load('ex1data1.txt'); % read comma separated data
X = data(:, 1); 
y = data(:, 2);

m = length(X) % number of training examples
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;
theta = gradientDescent(X, y, theta, alpha, iterations);
plot(X(:,2), X*theta, '-')
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);

