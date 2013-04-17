function P = get_softmax_generic(Mu, x, beta_x)
% x MxD data matrix
% Mu: KxD matrix of center 
% P: KxM matrix of softmax evaluation
% P is formed such that sum_i P(i,j) = 1, i.e. each column is normalized

%% IT WAS LIKE THIS BEFORE
% C = size(Mu,1); % Number of classes
% maxcut = log(realmax) - log(C); % Prevents overflow and underflow
% mincut = log(realmin);
% a = - beta_x * sq_dist(Mu',x');
% a = min(a, maxcut);
% a = max(a, mincut);
% %val  =  exp(a);
% %P = val./repmat(sum(val,1),C,1);
% % 
% % Repmat seems to be very slow
% P =  exp(a);
% Z = sum(P,1);
% for i = 1 : size(P,1)
%     P(i,:) = P(i,:)./Z;
% end
% % Ensure that log(P) is computable
% P(P<realmin) = realmin;
% 05/10/2011: Modified to accept beta_x as a vector
% beta_x can be a 1xK dimensional vector

K = size(Mu,1); % Number of centers
M = size(x,1);

Lbeta = length(beta_x); % Lbeta = K

if ( Lbeta > 1) 
    a = repmat(-beta_x', 1, M).* sq_dist(Mu',x');
else
    a = - beta_x * sq_dist(Mu',x');
end

      if ( any(isnan(a(:))) )
           disp('get_softmax_generic::NaN Value found');
      end

P = get_probabilities_softmax(a', K)';

return;







