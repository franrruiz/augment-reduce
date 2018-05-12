function negsamples = sample_negatives(K,B,ns,batch_idx,Y,flag_mexFile,sampling_weights)

if(nargin<6)
    flag_mexFile = 0;
end
if(nargin<7)
    sampling_weights = [];
end

% Negative samples
if(~flag_mexFile)
    negsamples = zeros(B,ns);
    for bb=1:B
        zn = Y(batch_idx(bb));
        if(isempty(sampling_weights))
            negsamples(bb,:) = randsample([1:zn-1 zn+1:K],ns);
        else
            negsamples(bb,:) = randsample([1:zn-1 zn+1:K],ns,true,sampling_weights([1:zn-1 zn+1:K]));
        end
    end
else
    negsamples = multirandperm(B,K,ns,Y(batch_idx),length(sampling_weights),sampling_weights,randi(intmax('int32'),1));
end
