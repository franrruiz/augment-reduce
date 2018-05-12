function [Phi_observed Phi_ns] = compute_psi_main(Xminibatch, negsamples, batch_idx, data, param, pvar)

if(param.flag_mexFile)
    [Phi_observed Phi_ns] = compute_psi(param.B, data.D, data.K, param.ns, Xminibatch, data.Y(batch_idx)-1, negsamples-1, pvar.weights.val, pvar.biases.val );
else
    Phi_observed = sum(Xminibatch.*pvar.weights.val(:,data.Y(batch_idx))',2) + pvar.biases.val(data.Y(batch_idx))';
    aux1 = permute(full(Xminibatch),[3 2 1]);
    aux2 = permute(reshape(pvar.weights.val(:,negsamples),[data.D param.B param.ns]),[1 3 2]);
    if(param.ns==1)
        Phi_ns = permute(tmult(aux1,aux2),[3 2 1]) + pvar.biases.val(negsamples)';
    else
        Phi_ns = permute(tmult(aux1,aux2),[3 2 1]) + pvar.biases.val(negsamples);
    end
end
