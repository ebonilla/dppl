% scales vector/matrix to have zero mean an unit variance (each column)
% x is matrix where each datapoint is a row
% u,dev: mean and std of training data
% if x is training data, please *dont* specify u,dev
% if x is test data, please specify u,dev
% function [x u dev] = normalise(x,u,dev) 
function [x u dev] = normalise(x,u,dev) 
[m n]=size(x);
if (nargin==1) % this is training data
  u = mean(x,1);
  dev = std(x,0,1); 
end
x = (x-repmat(u,m,1))./repmat(dev,m,1);


  
