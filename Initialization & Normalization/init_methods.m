function label=init_methods(data, K, choose)
% Initialization methods
% Input: data, the number of cluster and the method
% Output: label of cluster
% Written by kailugaji. (wangrongrong1996@126.com)
if choose==1
    % random
    [X_num, ~]=size(data);
    rand_array=randperm(X_num);   
    para_miu=data(rand_array(1:K), :); 
    % (X-para_miu)^2=X^2+para_miu^2-2*X*para_miu'. X_num*K
    distant=repmat(sum(data.*data,2),1,K)+repmat(sum(para_miu.*para_miu,2)',X_num,1)-2*data*para_miu';
    [~,label]=min(distant,[],2);
elseif choose==2
    % K-means
    label=kmeans(data, K);
elseif choose==3
    % fuzzy c-means
    options=[NaN, NaN, NaN, 0];
    [~, responsivity]=fcm(data, K, options);
    [~, label]=max(responsivity', [], 2);
elseif choose==4
    % K-means clustering, accelerated by matlab matrix operations.
    label = litekmeans(data, K,'Replicates',20);
end
