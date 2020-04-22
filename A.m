data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);
function g = sigmoid(z)
  g = zeros(size(z));
  g = 1./(1+exp(-z));
end

function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


sum = 0;
for i=1:m
    sum = sum + (-y(i)*(log(sigmoid(X(i,:)*theta)))-(1-y(i))*(log(1-(sigmoid(X(i,:)*theta)))));
end
J = (1/m)*sum;

  z = X * theta;      % m x 1
  h_x = sigmoid(z);   % m x 1 
  

grad = (1/m)* (X'*(h_x-y)); 

end

function p = predict(theta, X)

m = size(X, 1); % Number of training examples

p = zeros(m, 1);


for i=1:m
    if (X(m,1)*theta(1))+(X(m,2)*theta(2))+(X(m,3)*theta(3)) >= 0
        p(m,1) = 1;
    else
        p(m,1) = 0;
    end
end




end

