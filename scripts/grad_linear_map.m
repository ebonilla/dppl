% Computes the gradient of the linear map wrt w at values x
function grad_f = grad_linear_map(w,x)
N = size(x,1);
grad_f = [x , ones(N,1)];


