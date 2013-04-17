function[theta_pred, ypred, zpred]= predict_dppl(C, Mu, beta_x, w, xtest, low_test, up_test)
% Makes predicitions on new datapoints for the
% Discriminative Probabilistic Prototype Learning (DPPL) model 
% of Bonilla and Robles-Kelly (ICML, 2012)
%
% Below, K is the number of prototypes and D is the input dimensionality
%
% INPUT:
%   - C: The number of classes 
%   - Mu:  (KxD) matrix of initial centers. 
%   - beta_x: initial inverse temperature parameter o prototype function
%   - w: (C(D+1)x1) Vector of all initial weights for softmax model 
% xtest: (ntest x D) Matrix of all training vectors stacked together 
% low_test: (Ntest x D) vector of indices indicating where the corresponding
%           input vectors of each test point start
% up_test: (Ntest x D) vector of indices indicating where the corresponding
%           input vectors of each test point end
% OUTPUT:
%   - theta_pred: Cell array of predictive probabilities for each data-point
%   - ypred:  (Ntest x 1) vector of predicted classes for each data-point
%
% Edwin V. Bonilla

%% gets new codebook with learned beta and Mu and makes predictions
zpred = get_soft_codebook(xtest, low_test, up_test, Mu, beta_x);

%% Predictions and measure accuracy and test likelihood
[theta_pred, ypred] = fwd_softmax_class(w, C, @linear_map, zpred);



return;

