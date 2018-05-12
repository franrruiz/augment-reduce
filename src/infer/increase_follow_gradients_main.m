function pvar = increase_follow_gradients_main(Xminibatch, auxDerivative, negsamples, batch_idx, sviFactor, data, param, pvar, hyper)

dlogp_dwneg = -bsxfun(@times,permute(auxDerivative,[1 3 2]),full(Xminibatch));
dlogp_dwpos = -sum(dlogp_dwneg,3);
dlogp_dbneg = -auxDerivative;
dlogp_dbpos = sum(auxDerivative,2);

weights_ss = ones(1,param.B);
weights_ss_neg = ones(1,param.B*param.ns);

mask_pos = sparse(data.Y(batch_idx),1:param.B,weights_ss,data.K,param.B);
weights_grad = (mask_pos*dlogp_dwpos)';
biases_grad = (mask_pos*dlogp_dbpos)';

mask_neg = sparse(negsamples(:),1:param.B*param.ns,weights_ss_neg,data.K,param.B*param.ns);
reshaped_w = reshape(permute(dlogp_dwneg,[1 3 2]),[param.B*param.ns,data.D]);
reshaped_b = reshape(dlogp_dbneg,[param.B*param.ns,1]);
weights_grad = weights_grad + (mask_neg*reshaped_w)';
biases_grad = biases_grad + (mask_neg*reshaped_b)';

% Scale gradients by sviFactor
weights_grad = sviFactor*weights_grad;
biases_grad = sviFactor*biases_grad;

% Add L2 regularization term
weights_grad = weights_grad - pvar.weights.val/hyper.s2w;
biases_grad = biases_grad - pvar.biases.val/hyper.s2b;

% Take gradient step
[pvar.weights.val pvar.weights.Gt] = take_grad_step(pvar.weights.val,pvar.weights.Gt,weights_grad,param);
[pvar.biases.val pvar.biases.Gt] = take_grad_step(pvar.biases.val,pvar.biases.Gt,biases_grad,param);
