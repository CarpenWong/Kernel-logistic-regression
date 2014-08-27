%% Kernel Logistic Regression
clear all;
close all;

%Number of classes
c = 3;     

n1 = 1000;
n2 = 1000;
n3 = 1000;
n  = n1 + n2 + n3;
%Training data
X1 = randn(2,n1) - [3;0]*ones(1,n1);
X2 = randn(2,n2) + [1;1.5]*ones(1,n2);
X3 = randn(2,n3) + [0;-4]*ones(1,n2);
X = [X1 X2 X3];

%Label Data
label = [ones(n1,1); 2*ones(n2,1); 3*ones(n3,1)];

%KLR parameter
delta   = 0.0;    
itrNewton = 3; 

%Kernel Computation
deg = 10;
tic
Ktrain = kernel_Gaussian(X,X,deg);
toc
%KLR training 
tic
V_KLR = klr_train(Ktrain,label,delta,itrNewton);
toc

%% Computing the Dicision boundary of KLR
[x,y] = meshgrid(-6:.2:6, 6:-.2:-6);
X_grid = [x(:) y(:)]';

Ktest = kernel_Gaussian(X, X_grid,deg);
prob = mlogistic(V_KLR*Ktest,c);

rate1 = zeros(size(prob,2),1);
rate2 = zeros(size(prob,2),1);
for ii = 1:size(prob,2)
    rate1(ii) = prob(1,ii) - prob(2,ii) - prob(3,ii);
    rate2(ii) = prob(2,ii) - prob(1,ii) - prob(3,ii);
end

C_KLR1 = zeros(size(x));
C_KLR2 = zeros(size(x));
for ii = 1:size(x,1)
    for jj = 1:size(x,2)
        C_KLR1(jj,ii) = rate1(jj + size(x,1)*(ii -1));
        C_KLR2(jj,ii) = rate2(jj + size(x,1)*(ii -1));
    end
end

%% Plot Decision boundary
figure; 
plot(X1(1,:), X1(2,:), 'ro'); hold on;
plot(X2(1,:), X2(2,:), 'x');
plot(X3(1,:), X3(2,:), 'ks');
contour(x,y,C_KLR1,[0 0],'k'); 
contour(x,y,C_KLR2,[0 0],'k');
hold off;
legend('Class1-tr', 'Class2-tr', 'Class3-tr');
axis([-6 6 -6 6]);
