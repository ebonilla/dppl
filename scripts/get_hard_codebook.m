% beta_x: patameter of softmax function: z_k = sum_{x \in S} g_k(x)
%        where g_k(x) \propto exp ( - beta_x |u_k - x |^2 )
%        which replaces hardmax of codeboox approach
% x:  is a cell of matrices one for every data point
% Mu: centers (KxD) matrix with each row being a center
%
% Edwin V. Bonilla
%
% 07/09/2011: Modified to accept matrix of all x instead of cell

function z = get_hard_codebook(all_x, low, up, Mu)
N = length(low);
K = size(Mu,1);
z = zeros(N,K); % codebook representation

codeword = 1 : K;
for i = 1 : N 
    x          = all_x(low(i):up(i), :);
    dist       = sq_dist(x',Mu');
    [val code] = min(dist, [], 2);
    
    z(i,:) = histc(code, codeword)';
    
    
        
    % Added on 31/05/2011
    N_x    = size(x,1);
    z(i,:) = z(i,:)/N_x; 
        
    
end


return;




 