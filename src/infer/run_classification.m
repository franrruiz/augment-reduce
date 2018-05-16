function run_classification(data, param)

%% Get data parameters
[data.N data.D] = size(data.X);
data.K = max(data.Y);
[data.test.N data.test.D] = size(data.test.X);
data.Xtr = data.X';

%% Set config parameters

% Method and batch sizes

if(strcmp(param.method_name, 'softmax_a&r'))
    param.method = 'sm_augm';
elseif(strcmp(param.method_name, 'probit_a&r'))
    param.method = 'probit_persistent';
elseif(strcmp(param.method_name, 'logistic_a&r'))
    param.method = 'logistic_persistent';
elseif(strcmp(param.method_name, 'ove'))
    param.method = 'ove';
elseif(strcmp(param.method_name, 'botev'))
    param.method = 'botev';
elseif(strcmp(param.method_name, 'softmax'))
    param.method = 'softmax';
else
    error(['Unknown method name: ' param.method_name]);
end

if(strcmp(param.method,'softmax'))
    param.ns = data.K-1;
    param.flag_imp_sampling = 0;
end

% Step size schedule
param.step_schedule = 'stan';
param.step_kappa = 0.1;
param.step_T = 1;

%% Set hyperparameters
hyper.s2w = param.s2prior;
hyper.s2b = param.s2prior;

%% Initialize parameters (weights and biases)
pvar.weights.val = 0.1*randn(data.D,data.K);
pvar.weights.Gt = zeros(data.D,data.K);
pvar.biases.val = 0.001*randn(1,data.K);
pvar.biases.Gt = zeros(1,data.K);

%% Auxiliary variables for importance sampling
if(param.flag_imp_sampling>0)
    [class_unique, aux_counts] = count_unique(data.Y);
    class_counts = zeros(1,data.K);
    class_counts(class_unique) = aux_counts;
    param.is.freq_power = class_counts.^0.75;
    param.is.sum_freq_power = sum(param.is.freq_power);
    param.is.log_freq_power = log(param.is.freq_power);
    param.is.log_sum_freq_power_minus1class = log(param.is.sum_freq_power-param.is.freq_power);
else
    param.is.freq_power = [];
end

%% Set SVI factor
if(param.flag_imp_sampling>0)
    sviFactorClasses = 1/param.ns;
else
    sviFactorClasses = (data.K-1)/param.ns;
end
sviFactorData = data.N/param.B;
sviFactor = sviFactorData*sviFactorClasses;

%% Initialize local parameters
if(strcmp(param.method,'sm_augm'))
    pvar.sm_augm.eta = ones(data.N,1);
elseif(strcmp(param.method,'probit_persistent') || strcmp(param.method,'logistic_persistent'))
    pvar.local.mu = zeros(data.N,1);
    pvar.local.Gt_mu = zeros(data.N,1);
    pvar.local.std = ones(data.N,1);
    pvar.local.u_std = log(exp(pvar.local.std)-1);
    pvar.local.Gt_u_std = zeros(data.N,1);
end

%% Inference loop
pvar.obj_all = zeros(1,param.maxIter);
pvar.llh_all = zeros(1,param.maxIter);
pvar.lprior_all = zeros(1,param.maxIter);
pvar.telapsed_all = zeros(1,param.maxIter);
param.batch.state = 1;
param.batch.perm = randperm(data.N);
param.epoch = 1;
for tt=1:param.maxIter
    % Keep track of running time
    t_start = tic;
    
    % Save iteration number and increase epoch
    param.step_it = tt;
    if(param.batch.state+param.B-1>data.N)
        param.epoch = param.epoch+1;
    end
    
    % Gradual decrease of the baseline learning rate
	if(mod(tt,2000)==0)
		param.step_eta = 0.9*param.step_eta;
	end

    % Display progress
    if(mod(tt,round(param.maxIter/40))==0)
        disp(['Iteration ' num2str(tt) '/' num2str(param.maxIter) ' (' num2str(100*tt/param.maxIter,'%.1f') '%) (epoch ' num2str(param.epoch) ')...']);
    end
    
    % Sample minibatch of data
    [batch_idx, param.batch.state, param.batch.perm] = takeNextBatch(data.N, param.B, param.batch.state, param.batch.perm);
    Xminibatch = data.Xtr(:,batch_idx)';
    
    % Negative samples
    negsamples = sample_negatives(data.K,param.B,param.ns,batch_idx,data.Y,param.flag_mexFile,param.is.freq_power);
    
    % If 'probit_persistent' or 'logistic_persistent', take a grad step wrt local parameters
    if(strcmp(param.method,'probit_persistent') || strcmp(param.method,'logistic_persistent'))
        paramAux.step_schedule = 'robins';
        paramAux.step_eta = 0.001;
        paramAux.step_it = param.epoch;
        paramAux.step_T = 1;
        paramAux.step_kappa = 0.9;
        
        % Sample u
        if(strcmp(param.method,'probit_persistent'))
            [auxU auxE] = sample_auxiliary_probit(param.B,1,[pvar.local.mu(batch_idx) pvar.local.std(batch_idx)]);
        elseif(strcmp(param.method,'logistic_persistent'))
            [auxU auxE] = sample_auxiliary_logistic(param.B,1,[pvar.local.mu(batch_idx) pvar.local.std(batch_idx)]);
        else
            error('This should not happen');
        end
        
        % Take grad step for nn parameters
        % 1a. dlogp_du
        [Phi_observed Phi_ns] = compute_psi_main(Xminibatch, negsamples, batch_idx, data, param, pvar);
        if(strcmp(param.method,'probit_persistent'))
            auxDerivative = dlogPhi(bsxfun(@minus, auxU + Phi_observed, Phi_ns));
            auxDerivative = scale_importance_weights(auxDerivative, negsamples, batch_idx, data, param, pvar);
            dlogp_du = -auxU + sviFactorClasses*sum(auxDerivative,2);
        elseif(strcmp(param.method,'logistic_persistent'))
            auxDerivative = sigmoid(bsxfun(@minus, Phi_ns, auxU + Phi_observed));
            auxDerivative = scale_importance_weights(auxDerivative, negsamples, batch_idx, data, param, pvar);
            dlogp_du = 1 - 2*sigmoid(auxU) + sviFactorClasses*sum(auxDerivative,2);
        else
            error('This should not happen');
        end
        % 2. du_duncSigma
        du_duncSigma = auxE .* (1-exp(-pvar.local.std(batch_idx)));
        % 3. Take grad step
        [pvar.local.mu(batch_idx) pvar.local.Gt_mu(batch_idx)] = take_grad_step(pvar.local.mu(batch_idx),pvar.local.Gt_mu(batch_idx),dlogp_du,paramAux);
        [pvar.local.u_std(batch_idx) pvar.local.Gt_u_std(batch_idx)] = take_grad_step(pvar.local.u_std(batch_idx),pvar.local.Gt_u_std(batch_idx),dlogp_du.*du_duncSigma,paramAux);
        pvar.local.std(batch_idx) = softplus(pvar.local.u_std(batch_idx));
        
    % If 'sm_augm', maximize w.r.t. local parameters
    elseif(strcmp(param.method,'sm_augm'))
        % Stepsize
        paramAux.rho = 0.05;
        paramAux.rho = (1+param.epoch)^(-0.9);

        % Compute target natural parameters
        [Phi_observed Phi_ns] = compute_psi_main(Xminibatch, negsamples, batch_idx, data, param, pvar);
        if(param.flag_imp_sampling==0)
            auxDiff = bsxfun(@minus, Phi_ns, Phi_observed);
        else
            aux = param.is.log_freq_power(negsamples);
            if(param.ns==1)
                aux = aux';
            end
            auxDiff = bsxfun(@minus, Phi_ns-aux, Phi_observed-param.is.log_sum_freq_power_minus1class(data.Y(batch_idx))');
        end
        % to prevent numerical instabilities
        pvar.sm_augm.auxMax = max(auxDiff,[],2);
        pvar.sm_augm.auxPos = zeros(size(pvar.sm_augm.auxMax));
        pvar.sm_augm.exponentials = exp(bsxfun(@minus,auxDiff,pvar.sm_augm.auxMax.*pvar.sm_augm.auxPos));
        new_eta = sviFactorClasses*sum(pvar.sm_augm.exponentials,2);

        % Take grad step
        pvar.sm_augm.eta(batch_idx) = (1-paramAux.rho)*pvar.sm_augm.eta(batch_idx) + paramAux.rho*new_eta;
    end
    
    % Sample u (only for '_persistent' methods)
    if(~isempty(strfind(param.method,'persistent')))
        if(strcmp(param.method,'probit_persistent'))
            auxU = sample_auxiliary_probit(param.B,1,[pvar.local.mu(batch_idx) pvar.local.std(batch_idx)]);
        elseif(strcmp(param.method,'logistic_persistent'))
            auxU = sample_auxiliary_logistic(param.B,1,[pvar.local.mu(batch_idx) pvar.local.std(batch_idx)]);
        else
            error('This should not happen');
        end
    end
    
    % Compute Phi
    if(~strcmp(param.method, 'sm_augm'))
        % These variables are already up-to-date for the 'sm_augm' approach
        [Phi_observed Phi_ns] = compute_psi_main(Xminibatch, negsamples, batch_idx, data, param, pvar);
    end
    
    % Initialize and compute the gradients
    if(strcmp(param.method,'probit_persistent') || strcmp(param.method,'ove') || strcmp(param.method,'botev') || strcmp(param.method,'sm_augm') || strcmp(param.method,'logistic_persistent'))
        auxP_botev = [];
        if(strcmp(param.method,'ove'))
            auxDerivative = (1-sigmoid(bsxfun(@minus, Phi_observed, Phi_ns)));
            auxDerivative = scale_importance_weights(auxDerivative, negsamples, batch_idx, data, param, pvar);
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
            auxDerivative = softmax([Phi_observed auxNS_botev], 2);
            auxP_botev = auxDerivative(:,1);
            auxDerivative = auxDerivative(:,2:end);
        elseif(strcmp(param.method,'probit_persistent'))
            auxDerivative = dlogPhi(bsxfun(@minus, auxU + Phi_observed, Phi_ns));
            auxDerivative = scale_importance_weights(auxDerivative, negsamples, batch_idx, data, param, pvar);
        elseif(strcmp(param.method,'sm_augm'))
            auxDerivative = bsxfun(@rdivide,pvar.sm_augm.exponentials,1+pvar.sm_augm.eta(batch_idx));
        elseif(strcmp(param.method,'logistic_persistent'))
            auxDerivative = sigmoid(bsxfun(@minus, Phi_ns, auxU + Phi_observed));
            auxDerivative = scale_importance_weights(auxDerivative, negsamples, batch_idx, data, param, pvar);
        else
            error('This should not happen');
        end
        % Increase gradients and take grad step
        if(param.flag_mexFile)
            increase_follow_gradients(param.B, data.D, data.K, param.ns, Xminibatch, data.Y(batch_idx)-1, negsamples-1, ...
                                      pvar.weights.val, pvar.biases.val, pvar.weights.Gt, pvar.biases.Gt, ...
                                      auxDerivative, auxP_botev, sviFactor, hyper.s2w, hyper.s2b, ...
                                      param.step_eta, param.step_kappa, param.step_T, param.step_it);
        else
            pvar = increase_follow_gradients_main(Xminibatch, auxDerivative, negsamples, batch_idx, sviFactor, data, param, pvar, hyper);
        end
    elseif(strcmp(param.method,'softmax'))
        sm_probs = softmax([Phi_observed Phi_ns],2);
        p_obs = sm_probs(:,1);
        p_ns = sm_probs(:,2:end);

        dlogp_dwneg = -bsxfun(@times,permute(p_ns,[1 3 2]),full(Xminibatch));
        dlogp_dbneg = -p_ns;
        dlogp_dwpos = bsxfun(@times,1-p_obs,full(Xminibatch));
        dlogp_dbpos = 1-p_obs;
        
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
    else
        error(['Unknown method: ' param.method]);
    end
    
    % Keep track of running time
    pvar.telapsed_all(tt) = toc(t_start);
    
    % Compute objective function
    if(data.N<=1e5 || mod(tt,50)==0)
        [pvar.obj_all(tt) pvar.llh_all(tt) pvar.lprior_all(tt)] = compute_objective_function(data,param,pvar,hyper,batch_idx,negsamples);
    end
end


%% Predictions
if(param.computePredTrain)
    pred.train = compute_predictions(data,param,pvar,data.K,1000);
end
pred.test = compute_predictions(data.test,param,pvar,data.K,1000);

%% Save
nameFile = [param.output_path '/results_' param.method_name '_nClasses' num2str(param.ns)];
if(param.flag_imp_sampling>0)
    nameFile = [nameFile '_impSamp'];
end
save([nameFile '.mat'], 'pvar', 'param', 'hyper', 'pred', '-v7.3');
