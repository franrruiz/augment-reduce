function y = softplus(x)

idxLarge = (x>5);
idxSmall = ~idxLarge;

y = zeros(size(x));
y(idxSmall) = log(1+exp(x(idxSmall)));
y(idxLarge) = x(idxLarge) + log(1+exp(-x(idxLarge)));
