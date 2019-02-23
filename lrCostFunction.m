function [J, grad] = lrCostFunction(theta, X, y, lambda)
% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));
h= sigmoid(X * theta)
lH=log(h);
lH1=log(1-h);
reg=lambda/(2*m)*(theta(2:end,:)'*theta(2:end,:));%regularization
J=-(1/m)*sum((y.*lH)+(1-y).*lH1)+reg;%cost fn
grad=X'*(h-y)*(1/m)+[0;lambda/m*theta(2:end,:)]; %zero to make dimensions
% match, gradient decend regularized as in previous ass.
%theta=theta-(alpha/m)*X'*(h-y);
grad=grad(:);
% =============================================================
end
