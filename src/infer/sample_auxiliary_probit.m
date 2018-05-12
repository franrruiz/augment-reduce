function [auxU auxE] = sample_auxiliary_probit(B,flag_amortizedInf,u_param)

auxE = randn([B 1]);
if(flag_amortizedInf)
    auxU = u_param(:,1) + u_param(:,2).*auxE;
else
    auxU = auxE;
end
