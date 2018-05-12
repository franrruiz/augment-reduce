function S = softmax(Y,dim) 
%function S = softmax(Y,dim) 
%
%

Y = bsxfun(@minus, Y, max(Y,[],dim)); 
Y = exp(Y); 
S = bsxfun(@rdivide, Y, sum(Y,dim)); 
