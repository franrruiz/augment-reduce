function y = dlogPhi(x)

y = zeros(size(x));
idx20 = (x<-20);
idxN20 = (~idx20);
y(idx20) = -x(idx20);
y(idxN20) = normpdf(x(idxN20))./normcdf(x(idxN20));
