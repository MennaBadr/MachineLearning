function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));
% Setup some useful variables
m = size(X, 1);
% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
a1 = [ones(m, 1) X]'; %'
z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2];
a3 = sigmoid(Theta2 * a2);
K = num_labels;
y_k = eye(num_labels);
cost = zeros(K,1);
for i=1:m
    cost =cost+(-y_k(:,y(i)).*log(a3(:,i))-(1 -y_k(:,y(i))).*log(1-a3(:,i)));
end
J = sum(cost)/m;
regularizationTerm =sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2));%from 2 to ignore bias
J = J + regularizationTerm*lambda/(2 * m);
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
Delta1_2 = zeros(size(Theta2));
Delta1_1 = zeros(size(Theta1));
for r=1:m
    a1 = [1 ; X(r,:)'];
    z2 = Theta1 *a1 ;
    a2 = sigmoid(z2);
    a2 = [1;a2];%bias
    z3 = Theta2 *a2;
    a3 = sigmoid(z3);
    delta_3 = a3 - y_k(:,y(r));%we have 3 layers and since it is backward 
    %propagation then we start from output layer and backwards to the 1st
    %layer
    delta_2 = (Theta2'* delta_3).*[1;sigmoidGradient(z2)]; %to adjust dimensions 0,1 does not matter
    delta_2 = delta_2(2:end,1);%removing bias
    Delta1_2 = Delta1_2+delta_3*a2';
    Delta1_1 = Delta1_1+delta_2*a1';
end
%Part 3: Implement regularization with the cost function and gradients.
Theta1_grad=Delta1_1/m+(lambda/m)*[zeros(size(Theta1,1),1),Theta1(:,2:end)];
Theta2_grad=Delta1_2/m +(lambda/m)*[zeros(size(Theta2,1),1),Theta2(:,2:end)];
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% =========================================================================












