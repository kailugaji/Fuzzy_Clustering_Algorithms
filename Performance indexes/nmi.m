function MIhat = nmi(A, B)
%NMI Normalized mutual information
% A, B: 1*N;
% http://en.wikipedia.org/wiki/Mutual_information
% http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
% 鍙傝��: https://www.cnblogs.com/ziqiao/archive/2011/12/13/2286273.html
% Example :  
% (http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html)
% A = [1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3];
% B = [1 2 1 1 1 1 1 2 2 2 2 3 1 1 3 3 3];
% A = label1; B = label2; label:N*1
% nmi(A,B) % ans =  0.3646
if length(A) ~= length(B)
    error('length( A ) must == length( B)');
end
N = length(A);
A_id = unique(A);
K_A = length(A_id);
B_id = unique(B);
K_B = length(B_id);
% Mutual information
A_occur = double (repmat( A, K_A, 1) == repmat( A_id', 1, N ));
B_occur = double (repmat( B, K_B, 1) == repmat( B_id', 1, N ));
AB_occur = A_occur * B_occur';
P_A= sum(A_occur') / N;
P_B = sum(B_occur') / N;
P_AB = AB_occur / N;
MImatrix = P_AB .* log(P_AB ./(P_A' * P_B)+eps);
MI = sum(MImatrix(:));
% Entropies
H_A = -sum(P_A .* log(P_A + eps),2);
H_B= -sum(P_B .* log(P_B + eps),2);
%Normalized Mutual information
MIhat = MI / sqrt(H_A*H_B);
% MIhat = 2 * MI / (H_A+H_B); another version of NMIend
