function yy = compute_predictions(data,param,pvar,K,Nsamples)

if(param.flag_mexFile)
    impSampMean = 5;
    impSampStd = 5;
    
    if(strcmp(param.method,'softmax') || strcmp(param.method,'ove') || strcmp(param.method,'botev') || strcmp(param.method,'sm_augm'))
        modelType = 1;
    elseif(~isempty(strfind(param.method,'logistic')))
        modelType = 2;
    elseif(~isempty(strfind(param.method,'probit')))
        modelType = 3;
    else
        error(['Unknown method: ' param.method]);
    end
    
    Xtr = data.X';
    [yy.llh yy.acc yy.llh_all] = compute_predictions_c(data.N, data.D, K, Xtr, data.Y-1, pvar.weights.val, pvar.biases.val, ...
                                                       modelType, Nsamples, impSampStd, impSampMean, randi(intmax('int32'),1));
	yy.llh_all(yy.llh_all<-26) = -26;
    yy.llh = mean(yy.llh_all);              
    yy.probs = [];
else
    N = size(data.X,1);
    inner_all = bsxfun(@plus,data.X*pvar.weights.val,pvar.biases.val);
    [valnul idxMax] = max(inner_all,[],2);
    yy.acc = sum(idxMax==data.Y)/N;

    %% Probabilities
    if(strcmp(param.method,'softmax') || strcmp(param.method,'ove') || strcmp(param.method,'botev') || strcmp(param.method,'sm_augm'))
        logits = inner_all;
        logprobs = bsxfun(@minus, logits, logsumexp(logits,2));
        yy.probs = exp(logprobs);
        idxAux = sub2ind([N K],1:N,data.Y');
        yy.llh_all = logprobs(idxAux);
        yy.llh = mean(yy.llh_all);
    elseif(~isempty(strfind(param.method,'logistic')))
        impSampStd = 2;
        impSampMean = zeros(N,1);
        inner_obs = zeros(N,1);
        inner_aux = zeros(N,K-1);
        for nn=1:N
            kk = data.Y(nn);
            impSampMean(nn) = fminsearch(@(uu)-logsigmoid(uu)-logsigmoid(-uu)-sum(logsigmoid(bsxfun(@minus,uu'+inner_all(nn,kk),inner_all(nn,[1:kk-1 kk+1:K]))),2)', 0);
            inner_obs(nn) = inner_all(nn,kk);
            inner_aux(nn,:) = inner_all(nn,[1:kk-1 kk+1:K]);
        end
        logits = zeros(N,Nsamples);
        for ss=1:Nsamples
            auxU = impSampMean + impSampStd*randn(N,1);
            inner_diff = bsxfun(@minus,auxU+inner_obs,inner_aux);
            logits(:,ss) = sum(logsigmoid(inner_diff),2) + logsigmoid(auxU) + logsigmoid(-auxU) + 0.5*log(2*pi*impSampStd^2) + 0.5*(auxU-impSampMean).^2/impSampStd^2;
        end
        yy.llh_all = logsumexp(logits,2)-log(Nsamples);
        yy.llh = mean(yy.llh_all);
        yy.probs = [];
    elseif(~isempty(strfind(param.method,'probit')))
        impSampStd = 2;
        impSampMean = zeros(N,1);
        inner_obs = zeros(N,1);
        inner_aux = zeros(N,K-1);
        for nn=1:N
            kk = data.Y(nn);
            impSampMean(nn) = fminsearch(@(uu)0.5*uu.^2 - sum(log(normcdf(bsxfun(@minus,uu'+inner_all(nn,kk),inner_all(nn,[1:kk-1 kk+1:K])))),2)', 0);
            inner_obs(nn) = inner_all(nn,kk);
            inner_aux(nn,:) = inner_all(nn,[1:kk-1 kk+1:K]);
        end
        logits = zeros(N,Nsamples);
        for ss=1:Nsamples
            auxU = impSampMean + impSampStd*randn(N,1);
            inner_diff = bsxfun(@minus,auxU+inner_obs,inner_aux);
            logits(:,ss) = sum(log(normcdf(inner_diff)),2) - 0.5*auxU.^2 + 0.5*(auxU-impSampMean).^2/impSampStd^2 + log(impSampStd);
        end
        yy.llh_all = logsumexp(logits,2)-log(Nsamples);
        yy.llh = mean(yy.llh_all);
        yy.probs = [];
    else
        error(['Unknown method: ' param.method]);
    end
end
