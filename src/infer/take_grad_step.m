function [v Gt] = take_grad_step(v,Gt,grad,param)

% Take gradient step
if(strcmp(param.step_schedule,'adagrad'))
    Gt = Gt+grad.^2;
    idxN0 = (Gt~=0);
    v(idxN0) = v(idxN0)+param.step_eta*grad(idxN0)./sqrt(Gt(idxN0));
elseif(strcmp(param.step_schedule,'robins'))
    eta = param.step_eta/((1+param.step_it/param.step_T)^param.step_kappa);
    v = v+eta*grad;
elseif(strcmp(param.step_schedule,'stan'))
    kappa = param.step_kappa;
    if(param.step_it==1)
        kappa = 1;
    end
    Gt = kappa*grad.^2+(1-kappa)*Gt;
    idxN0 = (Gt~=0);
    v(idxN0) = v(idxN0)+param.step_eta*(param.step_it^(-0.5+1e-16))*grad(idxN0)./(param.step_T+sqrt(Gt(idxN0)));
elseif(strcmp(param.step_schedule,'rmsprop'))
    kappa = param.step_kappa;
    if(param.step_it==1)
        kappa = 1;
    end
    Gt = kappa*grad.^2+(1-kappa)*Gt;
    idxN0 = (Gt~=0);
    v(idxN0) = v(idxN0)+param.step_eta*grad(idxN0)./sqrt(Gt(idxN0));
elseif(strcmp(param.step_schedule,'constant'))
    v = v+param.step_eta*grad;
elseif(strcmp(param.step_schedule,'indiv-robins'))
    eta = param.step_eta/((1+param.step_it/param.step_T)^param.step_kappa);
    idxN0 = (grad~=0);
    v(idxN0) = v(idxN0)+eta*grad(idxN0)./abs(grad(idxN0));
elseif(strcmp(param.step_schedule,'indiv-constant'))
    idxN0 = (grad~=0);
    v(idxN0) = v(idxN0)+param.step_eta*grad(idxN0)./abs(grad(idxN0));
else
    error(['Unknown stepsize schedule: ' param.step_schedule]);
end
