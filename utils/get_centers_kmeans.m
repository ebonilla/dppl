function [centers z mse] = get_centers_kmeans(xtrain, K, niter, tol, centers, verbose)
%% function centers = get_centers(xtrain)
% 12/10/2011 Modified to also return cluster number if required
% z is the cluster number
% mse: the mean square error
%
% Edwin V. Bonilla

if (nargin==5)
    verbose = -1;
end

Ntrain = size(xtrain,1);
fprintf('Doing kmeans on training data ...\n');
options(1) = verbose; 
options(2) = tol; 
options(3) = tol; 
options(14) = niter;

% initialization 
if (isempty(centers))
    idx_centers = randperm(Ntrain);
    centers = xtrain(idx_centers(1:K),:);
end
if (nargout >= 2)
    [centers, options, post, sse, iter] = mykmeans(centers, xtrain, options);
    [val z] = max(post, [], 2);
    mse = sse(iter)/Ntrain;
else
    centers = mykmeans(centers, xtrain, options);
end

fprintf('kmeans done\n');
return;



