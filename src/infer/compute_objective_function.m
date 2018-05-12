function [obj llh lprior] = compute_objective_function(data,param,pvar,hyper,batch_idx,negsamples)

if(param.flag_imp_sampling>0)
    sviFactorClasses = 1/param.ns;
else
    sviFactorClasses = (data.K-1)/param.ns;
end

lprior = sum(sum(-0.5*log(2*pi) - 0.5*pvar.weights.val.^2/hyper.s2w)) ...
         + sum(sum(-0.5*log(2*pi) - 0.5*pvar.biases.val.^2/hyper.s2b));

if(strcmp(param.method,'softmax'))
    logits = bsxfun(@plus,data.X(batch_idx,:)*pvar.weights.val,pvar.biases.val);
    probs = softmax(logits,2);
    idxAux = sub2ind([param.B data.K],1:param.B,data.Y(batch_idx)');
    llh = (data.N/param.B)*sum(log(probs(idxAux)));
elseif(strcmp(param.method,'ove') || strcmp(param.method,'botev') || strcmp(param.method,'sm_augm'))
    [Phi_observed Phi_ns] = compute_psi_main(data.Xtr(:,batch_idx)', negsamples, batch_idx, data, param, pvar);
    if(strcmp(param.method,'ove'))
        aux = logsigmoid(bsxfun(@minus,Phi_observed,Phi_ns));
        aux = sviFactorClasses * scale_importance_weights(aux, negsamples, batch_idx, data, param, pvar);
        llh = (data.N/param.B)*sum(sum(aux));
    elseif(strcmp(param.method,'botev'))
        if(param.flag_imp_sampling>0)
            aux = param.is.log_freq_power(negsamples);
            if(param.ns==1)
                aux = aux';
            end
            auxNS_botev = bsxfun(@plus, Phi_ns - aux, param.is.log_sum_freq_power_minus1class(data.Y(batch_idx))') - log(param.ns);
        else
            auxNS_botev = Phi_ns + log((data.K-1)/param.ns);
        end
        aux = [Phi_observed auxNS_botev];
        llh = (data.N/param.B) * sum( aux(:,1) - logsumexp(aux,2) );
    elseif(strcmp(param.method,'sm_augm'))
        %llh = (data.N/param.B)*sum(1 - log(1+exp(pvar.sm_augm.auxMax.*pvar.sm_augm.auxPos).*pvar.sm_augm.eta(batch_idx)));
        eta_n = 1+exp(pvar.sm_augm.auxMax.*pvar.sm_augm.auxPos).*pvar.sm_augm.eta(batch_idx);
        mu_n = log(eta_n);
        aux = exp(-bsxfun(@minus,Phi_observed,Phi_ns));
        aux = sviFactorClasses * scale_importance_weights(aux, negsamples, batch_idx, data, param, pvar);
        llh = (data.N/param.B)*sum( -mu_n + 1 - (1 + sum(aux,2))./eta_n );
        %llh = (data.N/param.B)*sum( - mu_n + 1 - (1+((data.K-1)/param.ns)*sum(exp(-bsxfun(@minus,Phi_observed,Phi_ns)),2))./eta_n );
    else
        error('This should not happen');
    end
elseif(~isempty(strfind(param.method,'persistent')))
    [Phi_observed Phi_ns] = compute_psi_main(data.Xtr(:,batch_idx)', negsamples, batch_idx, data, param, pvar);
    if(strcmp(param.method,'logistic_persistent'))
        auxU = sample_auxiliary_logistic(param.B,1,[pvar.local.mu(batch_idx) pvar.local.std(batch_idx)]);
        aux = logsigmoid(bsxfun(@minus,auxU+Phi_observed,Phi_ns));
        aux = sviFactorClasses * scale_importance_weights(aux, negsamples, batch_idx, data, param, pvar);
        llh = (data.N/param.B)*sum( logsigmoid(auxU) + logsigmoid(-auxU) + sum(aux,2) + log(pvar.local.std(batch_idx)) + 2 );
        %llh = (data.N/param.B)*sum(logsigmoid(auxU) + logsigmoid(-auxU) + ((data.K-1)/param.ns)*sum(logsigmoid(bsxfun(@minus,auxU+Phi_observed,Phi_ns)),2) + log(pvar.local.std(batch_idx)) + 2);
    elseif(strcmp(param.method,'probit_persistent'))
        auxU = sample_auxiliary_probit(param.B,1,[pvar.local.mu(batch_idx) pvar.local.std(batch_idx)]);
        llh = (data.N/param.B)*sum(log(normpdf(auxU)) + ((data.K-1)/param.ns)*sum(log(normcdf(bsxfun(@minus,auxU+Phi_observed,Phi_ns))),2) + 0.5*log(2*pi*exp(1)*pvar.local.std(batch_idx).^2));
    else
        error(['Unknown method: ' param.method]);
    end
else
    error(['Unknown method: ' param.method]);
end

obj = llh + lprior;
