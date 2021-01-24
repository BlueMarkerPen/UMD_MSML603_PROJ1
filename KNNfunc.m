
function [solution] = KNNfunc(D, DT, L, LT, k) %train data, test dat, train label, test label, k's nearest neigher#class
n = size(D,2); % # training data
nt = size(DT,2); % # test data

%knn
%find the max value in the knn array, if the dist found between
%train data and test data less than the max value, replace the 
%max value to the dist, save the test label to knn_label
%then find the max # of label, set solution(i) equal to it.

solution = zeros(nt,1);
knn_data_index = [];
for i=1:nt

   for l=1:k
       knn(l)=10^10;  %this number has to be very large, o.w. no knn_data_laber
                      %can have some trouble when generalize this function
                      %this KNN alghrithm could be improved
       knn_data_index(l)=0;
   end
   
   for j=1:n
      [max_knn,max_index]=max(knn);
      dist =  (DT(:,i) - D(:,j))'*(DT(:,i) - D(:,j));
      %find the kth nearest dist and index.
      if dist<max_knn
          knn(max_index)=dist;
          knn_data_laber(max_index)=L(j);
      end
      solution(i) = mode(knn_data_laber);
   end
end

accuracy = 0.0;
for i=1:nt
   if solution(i) == LT(i)
       accuracy = accuracy + 1;
   end
end
accuracy = accuracy / nt;
display(accuracy);
