function [label,iter_FCM, para_miu, NegativeLogLikelihood, responsivity]=FCM_kailugaji(data, K, label_old, m)
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
count=zeros(data_num,1);  
responsivity=zeros(data_num,K);
para_miu=zeros(K, data_dim);
R_up=zeros(data_num,K);
%% initializing the cluster center
for k=1:K
    X_k=data(label_old==k, :); 
    para_miu(k, :)=mean(X_k);   % the center of each cluster
end
%% Fuzzy c-means algorithm
for t=1:max_iter
    % (X-para_miu)^2=X^2+para_miu^2-2*para_miu*X'. data_num*K
    distant=(sum(data.*data,2))*ones(1,K)+ones(data_num,1)*(sum(para_miu.*para_miu,2))'-2*data*para_miu';
    % update membership. data_num*K
    for i=1:data_num
        count(i)=sum(distant(i,:)==0);
        if count(i)>0
            for k=1:K
                if distant(i,k)==0
                    responsivity(i,k)=1./count(i);
                else
                    responsivity(i,k)=0;
                end
            end
        else
            R_up(i,:)=distant(i,:).^(-1/(m-1));  
            responsivity(i,:)= R_up(i,:)./sum( R_up(i,:),2); % membership
        end
    end
    % update center. K*data_dim
    miu_up=(responsivity'.^(m))*data;  
    para_miu=miu_up./((sum(responsivity.^(m)))'*ones(1,data_dim));
    % object function
    fitness(t)=sum(sum(distant.*(responsivity.^(m))));
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
