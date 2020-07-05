function data = normlization(data, choose)
% Normlization methods
% Written by kailugaji. (wangrongrong1996@126.com)
if choose==0
    % no normlization
    data = data;
elseif choose==1
    % Z-score
    data = bsxfun(@minus, data, mean(data));
    data = bsxfun(@rdivide, data, std(data));
elseif choose==2
    % max-min
    [data_num,~]=size(data);
    data=(data-ones(data_num,1)*min(data))./(ones(data_num,1)*(max(data)-min(data)));
end
