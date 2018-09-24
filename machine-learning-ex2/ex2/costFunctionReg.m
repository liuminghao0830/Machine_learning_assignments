function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute gradient of cost function
% for j = 1, no lamda involved
for i = 1:m
    h_theta = 1/(1 + exp(-theta'* X(i,:)'));
    grad(1) = grad(1) + (h_theta - y(i))* X(i,1);
end
grad(1) = grad(1)/m;

% for j = 2 ...
for j = 2:length(theta)
    for i = 1:m
        h_theta = 1/(1 + exp(-theta'* X(i,:)'));
        grad(j) = grad(j) + (h_theta - y(i))* X(i,j);
    end
    grad(j) = grad(j)/m + lambda/m*theta(j);
end

% Compute cost function
for i = 1:m
    h_theta = 1/(1 + exp(-theta'* X(i,:)'));
    J = J + (-y(i)*log(h_theta) - (1-y(i))*log(1-h_theta));
end

for j = 2:length(theta)
    J = J + 0.5*lambda * theta(j)^2;
end
J = J/m;


% =============================================================

end
