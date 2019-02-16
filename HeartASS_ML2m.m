clc
clear all
close all
ds =datastore('heart_DD.csv','TreatAsMissing','NA','MissingValue',0,'readsize',250);
T = read(ds);
size(T);

Alpha=.01;
%;
m=length(T{:,1});
U0=T{:,2};
U=T{:,1:13};

%U1=T{:,20:21};
%U2=U.^2;
X=[ones(m,1) U U.^2]; %U1 U.^2 U.^3];

n=length(X(1,:));
for w=2:n
    if max(abs(X(:,w)))~=0;
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
   
    end
end
lamda=0.0000100;


Y=T{:,14}/mean(T{:,14});
Theta=zeros(n,1);
k=1;

for j=1:length(Theta)
ee=exp(-X*Theta);
h=1./(1+(ee));
loggX=log(h);
loggX2=log(1-h);
g(j)=(1/m)*sum((h-Y)'*X(:,j));

Theta=Theta-(Alpha/m)*X'*(h-Y);
E(k)=-(1/m)*sum((Y.*loggX)+(1-Y).*loggX2)+(lamda/(2*m))*sum(Theta.^2);
%ERROR, ALL DATA WITHOUT DIVIDING THEM
end
R=1;


%%%%%%%%%TRAINING, Cross Validation, TEST%%%%%%%%%
mtrain=150;
mtest=(250-mtrain)/2;
mCV=(250-mtrain)/2;

Alpha1=0.001;


U_trainSET=T{1:mtrain,1:13};
UCV=T{mtrain+1:mtrain+mCV,1:13};
U_testSET=T{mtrain+mCV+1:end,1:13};
%U1_testSET=T{mCV+1:end,20:21};




s=1;
P=1;
lamda2=1900;
X1=[ones(mtrain,1) U_trainSET U_trainSET.^2]; %U_trainSET.^3 U_trainSET.^4 ];
X2=[ones(mtest,1) U_testSET U_testSET.^2]; %U_testSET.^3  U_testSET.^4];
X3=[ones(mCV,1) UCV UCV.^2]; %UCV.^3 UCV.^4];
%DIFFERENT HYPOTHESIS
%ALSO DIFFERENT COLUMNS DIFFERENT FEATURES COULD BE ADDED
%1ST DEGREE COUNTED AS 1ST HYPOTHESIS
%2ND DEGREE COUNTED AS 2ND HYPOTHESIS
%3RD DEGREE COUNTED AS 3RD HYPOTHESIS
%4TH DEGREE COUNTED AS 4TH HYPOTHESIS
n1=length(X1(1,:));
n2=length(X2(1,:));
n3=length(X3(1,:));

Theta1=zeros(n1,1);
Theta2=zeros(n2,1);
Theta3=zeros(n3,1);

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

YtrainSET=T{1:mtrain,14};%/mean(T{1:mtrain,14});
YCV=T{mtrain+1:mtrain+mCV,14};%/mean(T{mtrain+1:mtrain+mCV,14});
YtestSET=T{mtrain+mCV+1:end,14};%/mean(T{mtrain+mCV+1:end,14});

for j=1:length(Theta1)
ee1=exp(-X1*Theta1);
h1=1./(1+(ee1));
loggX1=log(h1);
loggX2=log(1-h1);
g1(j)=(1/m)*sum((h1-YtrainSET)'*X1(:,j));
k=k+1;
Theta1=Theta1-(Alpha/m)*X1'*(h1-YtrainSET);
ETrain(k)=-(1/m)*sum((YtrainSET.*loggX1)+(1-YtrainSET).*loggX2)+(lamda2/(2*m))*sum(Theta1.^2);

end
k=1;
for j=1:length(Theta)
ee2=exp(-X2*Theta);
h2=1./(1+(ee2));
loggX22=log(h2);
loggX2=log(1-h2);
g2(j)=(1/m)*sum((h2-YCV)'*X2(:,j));

Theta=Theta-(Alpha/m)*X2'*(h2-YCV);
Ecv(k)=-(1/m)*sum((YCV.*loggX22)+(1-YCV).*loggX2)+(lamda2/(2*m))*sum(Theta.^2);
k=k+1;
end
k=1;
for j=1:length(Theta)
ee3=exp(-X3*Theta);
h3=1./(1+(ee3));
loggX33=log(h3);
loggX333=log(1-h3);
g3(j)=(1/m)*sum((h3-YtestSET)'*X3(:,j));

Theta=Theta-(Alpha/m)*X3'*(h3-YtestSET);
ETest(k)=-(1/m)*sum((YtestSET.*loggX33)+(1-YtestSET).*loggX333)+(lamda2/(2*m))*sum(Theta.^2);
k=k+1;
end


plot(ETrain,'k','Linewidth',1.5)
hold on
plot(Ecv,'b','Linewidth',2.5)
hold on
plot(ETest,'--r','Linewidth',1.5)
legend('Trian','CV','Test')
title('Error')
%They are all single values but i made several single errros to plot them
%against each other.