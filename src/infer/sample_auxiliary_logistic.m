function [auxU auxE] = sample_auxiliary_logistic(B,flag_amortizedInf,u_param)

ee = random('exp',1,[B 1]);
auxE = ee+log(1-exp(-ee));
if(flag_amortizedInf)
    auxU = u_param(:,1) + u_param(:,2).*auxE;
else
    auxU = auxE;
end
