function z = get_soft_codebook(all_x, low, up, Mu, beta_x)
% Gets the (soft) codebook representation for all data in all_x
%
% INPUT:
%   - all_x: (n x D) Matrix of all training vectors stacked together
%   - low: (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point start
%   - up:  (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point end
%   - beta_x: initial inverse temperature parameter o prototype function
% OUTPUT:
% z: The (soft) codebook representation
%
% Edwin V. Bonilla

N = length(low);
K = size(Mu,1);
z = zeros(N,K); % codebook representation

for i = 1 : N
    X = all_x(low(i):up(i),:);
    
    % I AM DEBUGGING HERE
    % z(i,:) = sum(get_softmax_generic(Mu, X, beta_x),2)';
    z(i,:) = sum(get_softmax_generic(Mu, X, beta_x),2)';

     
%      if ( any(isnan(z(:))) )
%           disp('pause');
%      end

    % Added on 28/05/2011
    N_x    = size(X,1);
    z(i,:) = z(i,:)/N_x; 
        
end
  