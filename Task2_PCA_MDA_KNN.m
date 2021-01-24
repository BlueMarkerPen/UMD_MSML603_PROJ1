clc;
clear; 

%%
%load pose.mat
load('./data/pose.mat');  %48*40*13*68, 13 poses, 68 subjects
d = 48*40; % # dimension
c = 68; % # class
ratio = 0.7; % percentage of data for training
ni = round(13*ratio); % # training data per subject
n = ni*68; % # training data
nt = (13-ni)*68; % # test data
D = zeros(d,n);
L = zeros(n,1);
DT = zeros(d,nt);
LT = zeros(nt,1);
for i=0:c-1
    for j=1:ni
        D(:, ni*i+j) = reshape(pose(:,:,j,i+1), [d,1]);
        L(ni*i+j) = i+1;
    end
    for j=1:13-ni
        DT(:, (13-ni)*i+j) = reshape(pose(:,:,ni+j,i+1), [d,1]);
        LT((13-ni)*i+j) = i+1; 
    end
end

%%
%====>uncommon section to load data
%Load illumination.mat
% load('./data/illumination.mat');  %1920*21*68, 21 illuminations, 68 subjects
% d = 1920; % # dimension
% c = 68; % # class
% ratio = 0.8; % percentage of data for training
% ni = round(21*ratio) ;% # training data per subject
% n = ni*68 ;% # training data
% nt = (21-ni)*68 ;% # test data
% D = zeros(d,n);
% L = zeros(n,1);
% DT = zeros(d,nt);
% LT = zeros(nt,1);
%     
% for i=0:c-1
%     for j=1:ni
%         D(:, ni*i+j) = reshape(illum(:,j,i+1), [d,1]);
%         L(ni*i+j) = i+1;
%     end
%     for j=1:21-ni
%         DT(:, (21-ni)*i+j) = reshape(illum(:,ni+j,i+1), [d,1]);
%         LT((21-ni)*i+j) = i+1; 
%     end
% end

%%
%PCA
[W,S,V] = svds(D,200);
Y_p = zeros(200, n);
YT_p = zeros(200, nt);
for i = 1:n
   Y_p(:, i) = W.' * D(:,i);
end
for i = 1:nt
    YT_p(:, i) = W.' * DT(:,i);
end

%%
%LDA reduce to 67
% mean
mu = zeros(200, c); 
for i=1:c
    mu(:, i) = (Y_p(:,2*i-1) + Y_p(:,2*i))/2;
end

mu_all = zeros(200, 1);
for i=1:n
    mu_all = mu_all + Y_p(:,i);
end
mu_all = 1/n * mu_all;

% Within scatter matrix
delta = 0.05;  %SW singularity
SW = zeros(200,200);
for i=1:c
    for j=1:2
        S = ( Y_p(:,2*(i-1)+j) - mu(:,i) ) * ( Y_p(:,2*(i-1)+j) - mu(:,i) ).';
    end
    S = S + delta * eye(200);
    SW = SW + S;
end

if(det(SW)==0)
    DISP('singular');
    pause;
end



% Between scatter matrix
SB = zeros(200,200);
for i=1:c
   SB = SB + 2 * ( mu(:,i) - mu_all ) * ( mu(:,i) - mu_all ).';  
end

[W_m,EV] = eigs(SB,SW, c-1);


Y = zeros(c-1, n);
YT = zeros(c-1, nt);
for i = 1:n
   Y(:, i) = W_m.' * Y_p(:,i);
end
for i = 1:nt
    YT(:, i) = W_m.' * YT_p(:,i);
end
%%
%KNN
%change k here
k = 1;
solution = KNNfunc(Y, YT, L, LT, k);