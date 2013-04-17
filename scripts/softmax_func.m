function p = softmax_func(ptr_func, W, x)
% function p = softmax_func(ptr_func, W, x)
% Computes the softmax functon 
% for a matrix of parameters W and a vector  of features x
% and evaluation function ptr_func
% p = exp(f(w,t)) /sum_k f(w,k)
% In the case of classification each row of the matrix W
% are the paramters of a class of size (D+1). 
% The numer of rows of W is C, where C is the number of classes 
%
% ptr_func: is a pointer to the evaluation function
% W: matrix of Cx(D+1)
% x: data-points of NxD
%
% Modifed on 07/09/2011 to control overflow in a better way and not to
% care about underflow


%% It was like this before
% C = size(W,1); % Number of classes
% a = feval(ptr_func,W,x);
% %
% % Prevents overflow and underflow
% maxcut = log(realmax) - log(C); % Ensure that sum(exp(a), 2) does not overflow
% mincut = log(realmin); % ensure that exp(a) > 0
% a = min(a, maxcut);
% a = max(a, mincut);
% val =  exp(a);
% p = val./repmat(sum(val,2),1,C);
%  p(p<realmin) = realmin; % Ensure that log(p) is computable
% 

%% This is tbe new code
F    = feval(ptr_func,W,x);
C    = size(F,2);  % Number of classes
p    = get_probabilities_softmax(F,C);

% y    = F - repmat(max(F, [], 2), 1, C);
% p    = exp(y);
% Z    = sum(p,2);
% p    = p./repmat(Z, 1, C);


return;




   
 


