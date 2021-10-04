% load data
load('data_train.mat')
load('label_train.mat')
load('data_test.mat')

% divide the train set
self_test_idx = 71:100; % reserve 300 to test
train_idx = setdiff([1:330], self_test_idx);

%%
% paramrters
goal = 0.05;
spread = 1.4;
neuron_max = 20;

data_train_allo = data_train(train_idx, :)';
data_train_label = label_train(train_idx, :)';
data_train_ts = data_train(self_test_idx, :)';
data_train_ts_gt = label_train(self_test_idx, :)';

% train with 300 samples
net = newrb(data_train_allo, data_train_label, goal, spread, neuron_max);%,DF);
% test
pre_res = sim(net, data_train_ts);

precision = length(find((data_train_ts_gt .* pre_res) > 0)) / length(pre_res);
fprintf("Presicion is %.2f%%\n", precision * 100);
%% test
test_res = sim(net, data_test');
% bipolar mapping
for i = 1:length(test_res)
    if test_res(i) < 0
        test_res(i) = -1;
    else
        test_res(i) = 1;
    end
end
figure
plot(test_res, '-r')
hold on
plot(test_res, '*b')
grid on
axis on
axis equal
%% visulize diff
for i = 1:length(pre_res)
    if pre_res(i) < 0
        pre_res(i) = -1;
    else
        pre_res(i) = 1;
    end
end
res_rbf_diff = pre_res - data_train_ts_gt;
plot(res_rbf_diff,'MarkerSize', 10, 'Marker','*', 'Color', [0.8500 0.3250 0.0980])
grid on
axis on
axis equal