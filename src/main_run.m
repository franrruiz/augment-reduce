clear all;
close all;
clc;

%% Choose dataset (uncomment only one)
dataset_name = 'mnist';
% dataset_name = 'bibtex';
% dataset_name = 'eurlex';
% dataset_name = 'omniglot';
% dataset_name = 'amazoncat';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Choose the paths
data_path = '../../augment-reduce-data/dat';     % Replace with the path to your dataset
output_path = './results';                   % Replace with the path to the output folder

% NOTE: You can download the data used in the A&R paper from this repo:
% https://bitbucket.org/franrruiz/augment-reduce-data/src
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Choose method (uncomment only one)
method_name = 'softmax_a&r';  % Sofmax A&R
% method_name = 'probit_a&r';   % Multinomial probit A&R
% method_name = 'logistic_a&r'; % Multinomial logistic A&R
% method_name = 'ove';          % One-vs-each bound [Titsias, 2016]
% method_name = 'botev';        % The approach by [Botev et al., 2017]
% method_name = 'softmax';      % Exact softmax
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Choose parameters
param.flag_mexFile = 1;       % Use compiled mex files?
param.s2prior = inf;          % Variance of the prior over the weights and intercepts (inf for maximum likelihood)
param.step_eta= 0.02;         % Learning rate for the M step
param.flag_imp_sampling = 0;  % Use importance sampling when sampling negative classes?
param.computePredTrain = 0;   % Compute predictions on training data? (it may be expensive)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Get the data and default parameters
[data param] = get_params_preprocess_data(param, dataset_name, data_path, method_name);

%% Add paths
addpath aux
addpath infer
if(~isdir(output_path))
    mkdir(output_path);
end

%% Seed
rand('seed',0);
randn('seed',0);

%% Run classification
% 
% INPUTS:
% 
% data                        a struct containing the data
%    data.X                   the training data (instances x dimensions). It MUST be in sparse matrix format (use the command sparse).
%    data.Y                   the training labels (instances x 1). Each element indicates the class (from 1, ..., K).
%    data.test                a struct containing the test data
%       data.test.X           the test data (test_instances x dimensions). It MUST be in sparse matrix format.
%       data.test.Y           the test labels (test_instances x 1). Each element indicates the class (from 1, ..., K).
% param                       a struct containing the parameters
%    param.flag_mexFile       use compiled mex files?
%    param.flag_imp_sampling  use importance sampling when sampling negative classes?
%    param.s2prior            variance of the prior over the weights and intercepts (inf for maximum likelihood)
%    param.step_eta           learning rate for the M step
%    param.computePredTrain   compute predictions on training data? (it may be expensive)
%    param.maxIter            maximum number of iterations
%    param.B                  minibatch size (observations)
%    param.ns                 minibatch size (classes)
%    param.method             which method to use {'sm_augm', 'probit_persistent', 'logistic_persistent', 'ove', 'botev', 'softmax'}

run_classification(data, param);

