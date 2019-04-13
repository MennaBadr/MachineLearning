clc
clear all
close all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T = read(ds);
[f o]=size(T);
x_input=T{:,4:21};
[m n]=size(x_input);
Corr_x = corr(x_input);
x_cov=cov(x_input);
K = 0;
Alpha=0.01;
lamda=0.01;

% Normalisation
for w=1:n
    if max(abs(x_input(:,w)))~=0;
        x_input(:,w)=(x_input(:,w)-mean((x_input(:,w))))./std(x_input(:,w));
        
    end
end
%}
%{
The diagonal of the matrix contains the covariance between each variable and itself.
The other values in the matrix represent the covariance between
the two variables; in this case, the remaining two values are the same given
that we are calculating the covariance for only two variables.
%}

[U S V] = svd(x_cov); %Returns the eigenvectors U, the eigenvalues diag. in S;
%use S to find K

alpha=0.5;
while (alpha>=0.001)
    K=K+1;
    lamdas(K,:)=sum(max(S(:,1:K)));
    lamdass=sum(max(S));
    alpha=1-lamdas./lamdass;
end
R=U(:, 1:K)'*(x_input)';
app_data=U(:,1:K)*R;
error=(1/m)*(sum(app_data-x_input').^2);
%%%LINEAR REGRESSION%%%%%%
%%%LINEAR REGRESSION%%%%%%
h=1;
Theta=zeros(n,1);
k=1;
Y=T{:,3}/mean(T{:,3});
E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2); %cost function
while h==1
    Alpha=Alpha*1;
    Theta=Theta-(Alpha/m)*app_data*(app_data'*Theta-Y);
    k=k+1;
    E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2);
    
    %Regularization
    Reg(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2)+(lamda/(2*m))*sum(Theta.^2);
    %
    if E(k-1)-E(k)<0;
        break
    end
    q=(E(k-1)-E(k))./E(k-1);
    if q <.000001;
        h=0;
    end
end
%}
K
fprintf('Read Value of K before Reduced K means.\n');
pause;

K=6;
R=R';
costFunction = zeros(1,6);

%%%%%%K_MEANS_CLUSTER ON THE REDUCED%%%%%%%% 
%Iterate 
for q = 1:5
    %initialize 
    centroids = zeros(m,n);
    initial_index = randperm(m);
    centroids = R(initial_index(1:q),:);
    oldCentroids = zeros(size(centroids));
    indices = zeros(size(R,1), 1);
    distance = zeros(m,q);
   doNotStop = true;
    iterations = 0;
    while(doNotStop)
        for i = 1:m
            for j = 1:q
                distance(i, j) = sum((R(i,:) - centroids(j, :)).^2);
            end
        end
        for i = 1:m
            indices(i) = find(distance(i,:)==min(distance(i,:)));
        end
        for i = 1 : q
            clustering = R(find(indices == i), :);
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
numberOFClusters = 1:6;

plot(numberOFClusters, costFunction);
%}

%{
%%%%%%ANOMALY DETECTION%%%%%%
Epsilon = 0.5;
anomalyCount=0;
mu = mean(x_input);
sigma = (m-1)/m*var(x_input);
for a=1:m
ccdf(a,:)=normcdf(x_input(a,:),mu,sigma);
if ccdf(a,:)<Epsilon
   anomalyCount=anomalyCount+1; 
end
end
plot (anomalyCount,'*')
%}
%{
randidx = randperm(size(x_input, 1));
% Take the first K examples as centroids
centroids = x_input(randidx(1:K), :);
idx = zeros(size(x_input,1), 1);
g= size(x_input,1);
%%%%
for k = 1:K
  num_k = 0;
  sum = zeros(n, 1);
  for i = 1:m
    if ( idx(i) == k )
      sum = sum + x_input(i, :)';
      num_k = num_k + 1;
    end
  end
  centroids(k, :) = (sum/num_k)';
end
%%%%
for i = 1:g
    distance_array = zeros(1,K);
    for j = 1:K
        distance_array(1,j) = sqrt(sum(power((x_input(i,:)-centroids(j,:)),2)));
    end
    [d, d_idx] = min(distance_array);
    idx(i,1) = d_idx;
end

for i = 1:K
    c_i = idx==i;
    n_i = sum(c_i);
    c_i_matrix = repmat(c_i,1,m);
    X_c_i = x_input.* c_i_matrix;
    centroids(i,:) = sum(X_c_i) ./ n_i;
end
%}

%{

for k = 1:K
    point_indeces = find(idx==k);
    centroids(k, :) = sum(x_input(point_indeces, :))./length(point_indeces);
end
for i = 1:K
    c_i =idx==i;
    n_i =sum(c_i);
    X_c_i =c_i*x_input;
    centroids(i,:)=sum(X_c_i)./ n_i;
end
%%%OPTIMIZATION%%%

for i=1:n
    for j=1:m
optimization=(1/n)*sum((x_input(m,n)-c_i).^2);
    end
end

plot(centroids, '*', 'Linewidth',2.5)

%}

