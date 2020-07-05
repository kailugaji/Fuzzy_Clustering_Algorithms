function [accuracy, RI, NMI]=performance_index(real_label,our_id)
% Performance indices for clustering
% Input:
% real_label: Groundtruth. N*1
% our_id: train label. N*1
% Output:
% accuracy
% RI: Rand index
% NMI: Normalized Mutual Information
% Written by kailugaji. (wangrongrong1996@126.com)
[accuracy, label_new]=label_map(real_label,our_id);
[~,RI]=RandIndex(real_label,label_new);
NMI=nmi(real_label',label_new');
