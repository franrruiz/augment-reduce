function [data param] = get_params_preprocess_data(param, dataset_name, data_path, method_name)

%% Load the data and set parameters for each dataset
% The batch size B must be a dividend of the number of training points
if(strcmp(dataset_name, 'mnist'))
    
    param.maxIter = 35000;          % Maximum number of iterations
    param.B = 500;                  % Batch size (observations)
    param.ns = 1;                   % Batch size (classes)
    
    % Load the data
    load([data_path '/data_mnist.mat']);
    
elseif(strcmp(dataset_name, 'bibtex'))
    
    param.maxIter = 5000;           % Maximum number of iterations
    param.B = 488;                  % Batch size (observations)
    param.ns = 20;                  % Batch size (classes)
    
    % Load the data
    load([data_path '/data_bibtex.mat'], 'bibtex_struct');
    
    % Create classification labels by keeping first label in the multiclass array
    data.X = sparse(bibtex_struct.ft_train');
    data.test.X = sparse(bibtex_struct.ft_test');
    [data.Y, data.test.Y, K, idxTsRm] = keep_first_label(bibtex_struct.lbl_train', bibtex_struct.lbl_test', param.flag_mexFile);
    if(~isempty(idxTsRm))
        data.test.X(idxTsRm,:) = [];
        data.test.Y(idxTsRm) = [];
    end
    clear bibtex_struct;
    
elseif(strcmp(dataset_name, 'eurlex'))
    
    param.maxIter = 100000;         % Maximum number of iterations
    param.B = 379;                  % Batch size (observations)
    param.ns = 50;                  % Batch size (classes)
    
    % Load the data
    load([data_path '/data_eurlex.mat'], 'eurlex_struct');
    
    % Normalize by max value to preserve sparsity
    data.X = full(eurlex_struct.ft_train');
    data.test.X = full(eurlex_struct.ft_test');
    maxX = full(max(data.X,[],1));
    data.X = sparse(bsxfun(@rdivide, data.X, maxX));
    data.test.X = sparse(bsxfun(@rdivide, data.test.X, maxX));

    % Create classification labels by keeping first label in the multiclass array
    [data.Y data.test.Y K idxTsRm] = keep_first_label(eurlex_struct.lbl_train', eurlex_struct.lbl_test', param.flag_mexFile);
    if(~isempty(idxTsRm))
        data.test.X(idxTsRm,:) = [];
        data.test.Y(idxTsRm) = [];
    end
    clear eurlex_struct;
    
elseif(strcmp(dataset_name, 'omniglot'))
    
    param.maxIter = 45000;          % Maximum number of iterations
    param.B = 541;                  % Batch size (observations)
    param.ns = 50;                  % Batch size (classes)
    
    % Load the data
    load([data_path '/data_omniglot_resized.mat'], 'omniglot_struct');
    
    data.X = omniglot_struct.X;
    data.Y = omniglot_struct.Y;
    data.test.X = omniglot_struct.test.X;
    data.test.Y = omniglot_struct.test.Y;
    clear omniglot_struct;
    
elseif(strcmp(dataset_name, 'amazoncat'))
    
    param.maxIter = 5970;           % Maximum number of iterations
    param.B = 1987;                 % Batch size (observations)
    param.ns = 60;                  % Batch size (classes)
    
    % Load the data
    load([data_path '/data_amazoncat.mat'], 'amazon_struct');
    
    % Normalize by max value to preserve sparsity
    data.X = sparse(amazon_struct.ft_train');
    data.test.X = sparse(amazon_struct.ft_test');
    maxX = full(max(data.X,[],1));
    idxN0 = (maxX~=0);
    data.X(:,idxN0) = sparse(bsxfun(@rdivide, data.X(:,idxN0), maxX(idxN0)));
    data.test.X(:,idxN0) = sparse(bsxfun(@rdivide, data.test.X(:,idxN0), maxX(idxN0)));

    % Create classification labels by keeping first label in the multiclass array
    [data.Y, data.test.Y, K, idxTsRm] = keep_first_label(amazon_struct.lbl_train', amazon_struct.lbl_test', param.flag_mexFile);
    if(~isempty(idxTsRm))
        data.test.X(idxTsRm,:) = [];
        data.test.Y(idxTsRm) = [];
    end    
    clear amazon_struct;
    
else
    
    error(['Unknown dataset name: ' dataset_name]);
    
end

%% Select the method
if(strcmp(method_name, 'softmax_a&r'))
    param.method = 'sm_augm';
elseif(strcmp(method_name, 'probit_a&r'))
    param.method = 'probit_persistent';
elseif(strcmp(method_name, 'logistic_a&r'))
    param.method = 'logistic_persistent';
elseif(strcmp(method_name, 'ove'))
    param.method = 'ove';
elseif(strcmp(method_name, 'botev'))
    param.method = 'botev';
elseif(strcmp(method_name, 'softmax'))
    param.method = 'softmax';
else
    error(['Unknown method name: ' method_name]);
end


