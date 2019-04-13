clc
clear all
close all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T = read(ds);
[f o]=size(T);
X=T{:,4:21};
[m n]=size(X);
K = 15;
% Normalisation
for w=1:n
    if max(abs(X(:,w)))~=0;
        X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
        
    end
end
costFunction = zeros(1,15);
%%%%%%K_MEANS_CLUSTER%%%%%%%% 
%Iterate 
for q = 1:5
    %initialize 
    centroids = zeros(m,n);
    initial_index = randperm(m);
    centroids = X(initial_index(1:q),:);
    oldCentroids = zeros(size(centroids));
    indices = zeros(size(X,1), 1);
    distance = zeros(m,q);
    doNotStop = true;
    iterations = 0;
    while(doNotStop)
        for i = 1:m
            for j = 1:q
                distance(i, j) = sum((X(i,:) - centroids(j, :)).^2);
            end
        end
        for i = 1:m
            indices(i) = find(distance(i,:)==min(distance(i,:)));
        end
        for i= 1:q
            clustering = X(find(indices == i), :);
            centroids(i, :) = mean(clustering);
            cost = 0; %costfunction
            for z = 1 : size(clustering,1)
                cost = cost + (1/m)*sum((clustering(z,:) - centroids(i,:)).^2);
            end
            costFunction(1,q) = cost;
        end
         if oldCentroids == centroids
            doNotStop = false;
         end
        oldCentroids = centroids;
        iterations = iterations + 1;
        end
    end
[ o ,K_Optimal] = min(costFunction);
numberOFClusters = 1:15;
plot(numberOFClusters, costFunction);

