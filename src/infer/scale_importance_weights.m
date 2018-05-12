function y = scale_importance_weights(x, negsamples, batch_idx, data, param, pvar)

if(param.flag_imp_sampling==0)
    y = x;
    return;
end

aux = param.is.freq_power(negsamples);
if(param.ns==1)
    aux = aux';
end
y = x.*bsxfun(@rdivide, param.is.sum_freq_power-param.is.freq_power(data.Y(batch_idx))', aux);
