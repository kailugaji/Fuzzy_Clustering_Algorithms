function [label,iter_FCM, para_miu, NegativeLogLikelihood, responsivity]=MEC_kailugaji(data, K, label_old, gama)
% Input:
% K: number of cluster
% data: dataset, N*D
% label_old: initializing label. N*1
% Output:
% label: results of cluster. N*1
% iter_FCM: iterations
% Written by kailugaji. (wangrongrong1996@126.com)
format long 
%% initializing parameters
esp=1e-6;  % stopping criterion for iteration
max_iter=100;  % maximum number of iterations 
fitness=zeros(max_iter,1);
[data_num,data_dim]=size(data);
distant=zeros(data_num, K);
responsivity=zeros(data_num,K);
para_miu=zeros(K, data_dim);
%% initializing the cluster center
for k=1:K
    X_k=data(label_old==k, :); 
    para_miu(k, :)=mean(X_k); % the center of each cluster  
end
%% Maximum entropy clustering algorithm
for t=1:max_iter
    % (X-para_miu)^2=X^2+para_miu^2-2*para_miu*X'. data_num*K
    for k=1:K
        distant(:,k)=sum((data-repmat(para_miu(k, :), data_num, 1)).^2,2); %N*1
    end
    % update membership. data_num*K
    R_up=exp(-distant./gama);  
    responsivity= R_up./repmat(sum(R_up,2), 1, K);
    % update center. K*data_dim
    miu_up=(responsivity')*data;  
    para_miu=miu_up./((sum(responsivity))'*ones(1,data_dim));
    % object function
    fitness(t)=sum(sum(responsivity.*distant))+gama.*sum(sum((responsivity.*log(responsivity+eps))));
    if t>1  
        if abs(fitness(t)-fitness(t-1))<esp
            break;
        end
    end
end
iter_FCM=t;  % iterations
NegativeLogLikelihood=fitness(iter_FCM);
%% clustering
[~,label]=max(responsivity,[],2);
