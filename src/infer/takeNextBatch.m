function [batch_idx, batch_state, perm] = takeNextBatch(N, batch_size, batch_state, perm)  

% take a minibatch 
if (batch_state+batch_size-1) <= N
    batch_idx = perm(batch_state:batch_state+batch_size-1);
    batch_state = batch_state+batch_size;
else
    batch_state = 1; 
    perm = randperm(N);
    batch_idx = perm(batch_state:batch_state+batch_size-1);
    batch_state = batch_state+batch_size;
end 