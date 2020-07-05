% Demo Iris.data
% Written by kailugaji. (wangrongrong1996@126.com)
clear
clc
%% Setting the hyper-parameters
choose_norm=2; % Normalization methods, 0: no normalization, 1: z-score, 2: max-min
init=4; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.
repeat_num=10; % Repeat the experiment repeat_num times
choose_algorithm=3; % Fuzzy clustering algorithms, 1: Fuzzy c-means clustering (FCM), 2: Fuzzy subspace clustering (FSC), 3: Maximum entropy clustering (MEC)
addpath(genpath('.'));
%% Load data
data_load=dlmread('.\iris.data');
data=data_load(:, 1:end-1);
real_label=data_load(:, end);
K=length(unique(real_label)); % number of cluster
[N, ~]=size(data);
label_old=zeros(N, repeat_num);
s_1=0; 
%% Initialization & Normalization
data = normlization(data, choose_norm);
for i=1:repeat_num
    label_old(:, i)=init_methods(data, K, init);
end
%% Repeat the experiment repeat_num times
t0=cputime;
for i=1:repeat_num
    if choose_algorithm==1
        m=2; % fuzzy index
        [label,iter_FCM]=FCM_kailugaji(data, K, label_old(:, i), m);
    elseif choose_algorithm==2
        tao=2; % an weighted index
        sigm=1e-5; % a weighted regularization parameter
        [label,iter_FCM]=FSC_kailugaji(data, K, label_old(:, i), tao, sigm);
    elseif choose_algorithm==3
        gama=0.3; % a regularization parameter
        [label,iter_FCM]=MEC_kailugaji(data, K, label_old(:, i), gama);
    end
    iter_FCM_t(i)=iter_FCM;
    %% performanc indices
     [accuracy(i), RI(i), NMI(i)]=performance_index(real_label,label);
     s_1=s_1+iter_FCM_t(i);
     fprintf('Iteration %2d, the number of iterations: %2d, Accuary: %.8f\n', i, iter_FCM_t(i), accuracy(i));
end
run_time=cputime-t0;
%% Calculating evaluation indexes
repeat_num=length(find(accuracy~=0));
ave_iter_FCM=s_1/repeat_num; 
ave_run_time=run_time/repeat_num;
ave_acc_FCM=mean(accuracy); max_acc_FCM=max(accuracy); min_acc_FCM=min(accuracy);std_acc_FCM=std(accuracy);
ave_RI_FCM=mean(RI); max_RI_FCM=max(RI); min_RI_FCM=min(RI);std_RI_FCM=std(RI);
ave_NMI_FCM=mean(NMI); max_NMI_FCM=max(NMI); min_NMI_FCM=min(NMI);std_NMI_FCM=std(NMI);
fprintf('The average iteration number of the algorithm is: %.2f\nThe average running time is: %.5f\nThe average accuracy is: %.8f\nThe average rand index is: %.8f\nThe average normalized mutual information is: %.8f\n', ave_iter_FCM, ave_run_time, ave_acc_FCM, ave_RI_FCM, ave_NMI_FCM);
ACC=[ave_acc_FCM; std_acc_FCM; max_acc_FCM; min_acc_FCM];
ARI=[ave_RI_FCM; std_RI_FCM; max_RI_FCM; min_RI_FCM];
ANMI=[ave_NMI_FCM; std_NMI_FCM; max_NMI_FCM; min_NMI_FCM];
performance_indices=[ACC; ARI; ANMI; ave_iter_FCM; ave_run_time];
% performance_indices: 
% ave_acc_FCM
% std_acc_FCM
% max_acc_FCM
% min_acc_FCM
% ave_RI_FCM
% std_RI_FCM
% max_RI_FCM
% min_RI_FCM
% ave_NMI_FCM
% std_NMI_FCM
% max_NMI_FCM
% min_NMI_FCM
% ave_iter_FCM
% ave_run_time
save FCM_results performance_indices
rmpath(genpath('.'));
