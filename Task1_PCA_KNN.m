data = load('./data/data.mat');
d = 24*21;
n = 300; % # of training data \
n_illum = 0; % # of extra illum training data
n_d = n/2; % # of data from neutral and smiling
nt = 400-n; % # test data
c = 2; % # class

face = data.face;
face_neutral = face(:,:,1:3:end);
face_exp = face(:,:,2:3:end);
face_illum = face(:,:,3:3:end);
D = zeros(d, n+n_illum);
DT = zeros(d, nt);

L = zeros(n+n_illum,1); % label for training data
LT = zeros(nt,1); % label for test data

for i=1:n_d
    D(:,i) = reshape(face_neutral(:,:,i),[d,1]);
    L(i) = 1;%neutral
end

for i=n_d:n
    count =1;
    D(:,i) = reshape(face_exp(:,:,count),[d,1]);
    L(i) = 2;%smile
    count = count+1;
end

for i=n:(n+n_illum)
    count = 1;
    D(:,i) = reshape(face_illum(:,:,count),[d,1]);
    count = count+1;
    L(i) = 1;%neutral
end

for i=1:nt
    if i<=(nt/2)
        DT(:,i) = reshape(face_neutral(:,:,i+n_d),[d,1]);
        LT(i) = 1;%neutral
    else
        DT(:,i) = reshape(face_exp(:,:,i+(n_d-nt/2)),[d,1]);
        LT(i) = 2;%smile
    end
end


%%
%PCA
[W,S,V] = svds(D,c-1);
Y = zeros(c-1, n);
YT = zeros(c-1, nt);
for i = 1:n
   Y(:, i) = W.' * D(:,i);
end
for i = 1:nt
    YT(:, i) = W.' * DT(:,i);
end
%%
%KNN
%change k for the KNN here: 1, 3, 5, 7, 9, 11, 13.....
k = 3;
solution = KNNfunc(Y, YT, L, LT,k);
