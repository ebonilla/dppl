function P = get_probabilities_softmax(F, C)
%
% F:  NxC matrix
% P:  maxtrix of probabilities such that sum_j p(i,j) = 1
%
%% IT WAS LIKE THIS BEFORE
% maxcut = log(realmax) - log(C);
% mincut = log(realmin);
% F = min(F, maxcut);
% F = max(F, mincut);
% val =  exp(F);
% p = val./repmat(sum(val,2),1,C);
% p(p<realmin) = realmin;% Ensure that log(p) is computable


% C    = size(F,2);  % Number of classes
y    = F - repmat(max(F, [], 2), 1, C);
P    = exp(y);
Z    = sum(P,2);
P    = P./repmat(Z, 1, C);


% if ( any(isnan(p(:))) )
%     disp('pause');
% end

return;


