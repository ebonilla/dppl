% beta_x: patameter of softmax function: z_k = sum_{x \in S} g_k(x)
%        where g_k(x) \propto exp ( - beta_x |u_k - x |^2 )
%        which replaces hardmax of codeboox approach
% x:  is a cell of matrices one for every data point
% Mu: centers (KxD) matrix with each row being a center
% size(cell_x{i},2)=size(Mu,2)
function z = get_hard_codebook_cell(cell_x, Mu)
N = length(cell_x);
K = size(Mu,1);
z = zeros(N,K); % codebook representation

codeword = 1 : K;
for i = 1 : N 
    x          = cell_x{i};
    dist       = sq_dist(x',Mu');
    [val code] = min(dist, [], 2);
    
    z(i,:) = histc(code, codeword)';
    
    
        
    % Added on 31/05/2011
    % The models were trained without using this
    % but I used it for the toy data
    N_x    = size(x,1);
    z(i,:) = z(i,:)/N_x; 
        
    
end


return;




