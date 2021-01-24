clc;
clear; 

load('./data/pose.mat');  %48*40*13*68, 13 poses, 68 subjects
d = 48*40; % # dimension
c = 5; % # class
ratio = 0.8; % percentage of data for training
ni = round(13*ratio); % ratio= 0.8, ni=10, first 10 images
n = ni*5; % # training data
nt = (13-ni)*5; % # test data
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
%PCA reduce to 20
[W,S,V] = svds(D,20);
Y_p = zeros(20, n);
YT_p = zeros(20, nt);
for i = 1:n
   Y_p(:,i) = W.' * D(:,i);
end
for i = 1:nt
    YT_p(:,i) = W.' * DT(:,i);
end

%%
%LDA reduce to 4
% mean
mu = zeros(20, c); 
for i=1:c
    mu(:, i) = (Y_p(:,2*i-1) + Y_p(:,2*i))/2;
end

mu_all = zeros(20, 1);
for i=1:n
    mu_all = mu_all + Y_p(:,i);
end
mu_all = 1/n * mu_all;

% Within scatter matrix
delta = 0.05;  %SW singularity
SW = zeros(20,20);
for i=1:c
    for j=1:2
        S = ( Y_p(:,2*(i-1)+j) - mu(:,i) ) * ( Y_p(:,2*(i-1)+j) - mu(:,i) ).';
    end
    S = S + delta * eye(20);
    SW = SW + S;
end

if(det(SW)==0)
    DISP('singular');
    pause;
end



% Between scatter matrix
SB = zeros(20,20);
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
%Bayes
delta = 1; %var singularity
solution = BAYESfunc(Y, YT, LT, c, delta);
