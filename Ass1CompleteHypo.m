clc
clear all
close all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
size(T);
Alpha=0.01;
lamda=0.001;
%;
m=length(T{:,1});
U0=T{:,2};
U=T{:,4:10};


U1=T{:,20:21};
U2=U.^2;
X=[ones(m,1) U U1 U.^2 U.^3]; %U.^4, U.^5 U.^6 % DIFFERENT HYPOTHESIS
%ALSO DIFFERENT COLUMNS DIFFERENT FEATURES COULD BE ADDED
%1ST DEGREE COUNTED AS 1ST HYPOTHESIS
%2NT DEGREE COUNTED AS 2ND HYPOTHESIS
%3RD DEGREE COUNTED AS 3RD HYPOTHESIS
%4TH DEGREE COUNTED AS 4TH HYPOTHESIS
n=length(X(1,:));
for w=2:n
    if max(abs(X(:,w)))~=0;
        X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
        
    end
end

Y=T{:,3}/mean(T{:,3});
Theta=zeros(n,1);
k=1;
E(k)=(1/(2*m))*sum((X*Theta-Y).^2); %cost function


R=1;

while R==1
    Alpha=Alpha*1;
    Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
    k=k+1;
    E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
    
    %Regularization
    Reg(k)=(1/(2*m))*sum((X*Theta-Y).^2)+(lamda/(2*m))*sum(Theta.^2);
    %
    if E(k-1)-E(k)<0;
        break
    end
    q=(E(k-1)-E(k))./E(k-1);
    if q <.000001;
        R=0;
    end
end
figure(1)
plot(E,'k','Linewidth',2.5)
hold on
plot(Reg,'--.r','Linewidth',0.5)
legend('Cost Fn', 'Regularization')



%%%%%%%%%TRAINING, Cross Validation, TEST%%%%%%%%%

mtrain=12965;
mtest=(21607-mtrain)/2;
mCV=(21607-mtrain)/2;

Alpha1=0.001;

U_trainSET=T{1:mtrain,4:6};
UCV=T{mtrain+1:mtrain+mCV,4:6};
U_testSET=T{mtrain+mCV+1:end,4:6};



P=1;
lamda2=0.000001;
%DIFFERENT HYPOTHESIS
%ALSO DIFFERENT COLUMNS DIFFERENT FEATURES COULD BE ADDED
%1ST DEGREE COUNTED AS 1ST HYPOTHESIS
%2NT DEGREE COUNTED AS 2ND HYPOTHESIS
%3RD DEGREE COUNTED AS 3RD HYPOTHESIS
%4TH DEGREE COUNTED AS 4TH HYPOTHESIS
X1=[ones(mtrain,1) U_trainSET];% U_trainSET.^2 U_trainSET.^3 U_trainSET.^4]; %% %U_trainSET.^5 ];
X2=[ones(mtest,1) U_testSET];% U_testSET.^2 U_testSET.^3 U_testSET.^4]; % % %U_testSET.^5 ];
X3=[ones(mCV,1) UCV];% UCV.^2 UCV.^3 UCV.^4]; % % %UCV.^4 %UCV.^5];

n1=length(X1(1,:));
n2=length(X2(1,:));
n3=length(X3(1,:));

Theta1=zeros(n1,1);
Theta2=zeros(n2,1);
Theta3=zeros(n3,1);
% Normalization
for w1=2:n1
    if max(abs(X1(:,w1)))~=0;
        X1(:,w1)=(X1(:,w1)-mean((X1(:,w1))))./std(X1(:,w1));
        
    end
end
for w2=2:n2
    if max(abs(X2(:,w2)))~=0;
        X2(:,w2)=(X2(:,w2)-mean((X2(:,w2))))./std(X2(:,w2));
        
    end
end
for w3=2:n3
    if max(abs(X3(:,w3)))~=0;
        X3(:,w3)=(X3(:,w3)-mean((X3(:,w3))))./std(X3(:,w3));
        
    end
end
YtrainSET=T{1:mtrain,3}/mean(T{1:mtrain,3});
YCV=T{mtrain+1:mtrain+mCV,3}/mean(T{mtrain+1:mtrain+mCV,3});
YtestSET=T{mtrain+mCV+1:end,3}/mean(T{mtrain+mCV+1:end,3});
%Regularization 

%ERRORS%
s=1;
%
for j=1:length(Theta1)
    Theta1=Theta1-(Alpha/mtrain)*X1'*(X1*Theta1-YtrainSET);
    E1(s)=(1/(2*mtrain))*sum((X1*Theta1-YtrainSET).^2)+(lamda2/(2*mtrain))*sum(Theta1.^2);
    s=s+1;
end
s=1;
for k=1:length(Theta2);
    Theta2=Theta2-(Alpha/mtest)*X2'*(X2*Theta2-YtestSET); %mCV=mTrian, same theta length

    E_test(s)=(1/(2*mtest))*sum((X2*Theta2-YtestSET).^2)+(lamda2/(2*mtest))*sum(Theta2.^2); %TEST ERROR IN WORKSPACE AS VALUE%%%%
   
    s=s+1;
end
s=1;
for k=1:length(Theta3);
    Theta3=Theta3-(Alpha/mCV)*X3'*(X3*Theta3-YCV);
    E_CV(s)=(1/(2*mCV))*sum((X3*Theta3-YCV).^2)+(lamda2/(2*mCV))*sum(Theta3.^2); %CrossVal ERROR IN WORKSPACE AS VALUE%%%%
    s=s+1;
end
%They are single values but i looped around them several times to see them
%plotted against each other.
figure()
plot(E1,'k','Linewidth',1.5)
hold on
plot(E_test,'b','Linewidth',2.5)
hold on
plot(E_CV,'--r','Linewidth',1.5)
legend('Trian','Test','CV')
title('Errors')

%values of train and CV are very near to each other 