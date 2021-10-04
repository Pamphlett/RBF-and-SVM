% load data
load('data_train.mat')
load('label_train.mat')
load('data_test.mat')

% divide the train set
self_test_idx = 71:100; % reserve 300 to test
train_idx = setdiff([1:330], self_test_idx);
%%
data_train_allo = data_train(train_idx, :);
data_train_label = label_train(train_idx, :);
data_train_ts = data_train(self_test_idx, :);
data_train_ts_gt = label_train(self_test_idx, :);

m_svm = fitcsvm(data_train_allo, data_train_label, 'Standardize', true, 'KernelFunction',...
            'RBF', 'KernelScale', 'auto');
        
% m_svm = fitcsvm(data_train_allo, data_train_label, 'KernelFunction',...
%             'RBF');

% m_svm = fitcsvm(data_train_allo,data_train_label, 'Standardize',true,'KernelFunction',...
%             'RBF','KernelScale','auto', 'OptimizeHyperparameters','auto', ...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
%     'expected-improvement-plus'));

pre_res_svm = predict(m_svm, data_train_ts);
precision = length(find((data_train_ts_gt .* pre_res_svm) > 0)) / length(pre_res_svm);
fprintf("Presicion is %.2f%%\n", precision * 100);

%%
test_res_svm = predict(m_svm, data_test);
figure
plot(test_res, '-r')
hold on
plot(test_res, '*b')
grid on
axis on
axis equal

%% visulize diff
res_svm_diff = pre_res_svm - data_train_ts_gt;
plot(res_svm_diff,'MarkerSize', 20, 'Marker','.')
grid on
axis on
axis equal