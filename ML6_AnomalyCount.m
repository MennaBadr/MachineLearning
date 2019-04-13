clc
clear all
close all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T = read(ds);
[f o]=size(T);
x_input=T{:,4:21};
[m n]=size(x_input);
for w=1:n
    if max(abs(x_input(:,w)))~=0;
        x_input(:,w)=(x_input(:,w)-mean((x_input(:,w))))./std(x_input(:,w));
    end
end
%%%%%%ANOMALY DETECTION%%%%%%
Epsilon = 0.5e-1;
mu = mean(x_input);
sigma = (m-1)/m*var(x_input);
AnomalyVec=zeros(1,n);
for a=1:m 
ccdf(a,:)=normcdf(x_input(a,:),mu,sigma);
end
for a=1:n
    anomaly=0;
   for  b=2:m
   multiply(b-1,a)=ccdf(b-1,a)*ccdf(b,a);
    if multiply(b-1,a)<Epsilon %|| multiply(b-1,a)<1-Epsilon
       anomaly=anomaly+1;
    end
    AnomalyVec(a)=anomaly;
   end
end