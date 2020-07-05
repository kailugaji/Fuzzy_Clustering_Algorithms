function [label,iter_FCM, para_miu, para_w, NegativeLogLikelihood, responsivity]=FSC_kailugaji(data, K, label_old, tao, sigm)
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
eps=1e-6;  % stopping criterion for iteration
max_iter=100;  % maximum number of iterations 
fitness=zeros(max_iter,1);
[data_num,data_dim]=size(data);
responsivity=zeros(data_num,K);
w_temp=zeros(K, data_dim);
distants=zeros(data_num, K);
para_miu=zeros(K, data_dim);
%% initializing membership
for i=1:data_num
    responsivity(i, label_old(i))=1;
end
%% Fuzzy subspace clustering algorithm
for t=1:max_iter
    % update center. K*data_dim
    miu_up=(responsivity')*data;  
    para_miu=miu_up./((sum(responsivity))'*ones(1,data_dim));
    % update weight matrix. K*data_dim
    for k=1:K
        w_temp(k, :)=sum(repmat(responsivity(:,k), 1, data_dim).*((data-repmat(para_miu(k,:), data_num, 1)).^2))+sigm.*ones(1, data_dim);  %1*D
    end
    w_up=w_temp.^(-1/(tao-1));  
    para_w=w_up./repmat(sum(w_up,2), 1, data_dim); % weight
    % update membership. data_num*K
    for k=1:K
        distants(:,k)=sum(repmat(para_w(k, :).^tao, data_num, 1).*((data-repmat(para_miu(k, :), data_num, 1)).^2), 2);
    end
    for i=1:data_num
        [~, index]=min(distants(i,:));
        for k=1:K
            if index==k
                responsivity(i,index)=1;
            else 
                responsivity(i,k)=0;
            end
        end
    end
    % object function
    fitness(t)=sum(sum(responsivity.*distants))+sigm*sum(sum(para_w.^tao));
    if t>1  
        if abs(fitness(t)-fitness(t-1))<eps
            break;
        end
    end
end
iter_FCM=t;  % iterations
NegativeLogLikelihood=fitness(iter_FCM);
%% clustering
[~,label]=max(responsivity,[],2);
