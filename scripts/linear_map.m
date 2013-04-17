% function linear_map(w,x)
% Given the vector of weights w and the features x, it computes 
% the linear map w'x.
% It assumes that w alredy contains the bias b so that
% w = [w_1, w_2, ...,w_D, b] and therefore x is augmented with one 
% x: NxD matrix of data-points
% inside the function 
% w can also be a matrix 
function f = linear_map(w,x)
[N D] = size(x); 
x = [x,ones(N,1)];
f = x*w';







