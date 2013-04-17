function [low, up] = get_cell_limit(cell_xtrain)
Ntrain = length(cell_xtrain);
v_ntrain  = zeros(Ntrain,1);
for i = 1 : Ntrain
    v_ntrain(i) = size(cell_xtrain{i},1);
end

v_ntrain  =  cumsum(v_ntrain);


low = zeros(Ntrain,1);
low(1) = 1;
up     = v_ntrain; 
low(2:Ntrain) = up(1:Ntrain-1)+1;

