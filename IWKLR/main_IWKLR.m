%% Importance Weighted KLR
%
clear all;
close all;

c = 2;     %Number of classes

n = 1000;   
X1 = randn(2,n) - [3;1]*ones(1,n);
X2 = randn(2,n) + [1.5;1.5]*ones(1,n);

X = [X1 X2];
label = [ones(n,1); 2*ones(n,1)];

Ktrain = kernel_Poly(X,X,1);

delta   = 0.01;     
itrNewton = 3;

%KLR training 
V_KLR = klr_train(Ktrain,label,delta,itrNewton);

%% Computing the Dicision boundary of KLR
[x,y] = meshgrid(-6:.2:6, 6:-.2:-6);
X_grid = [x(:) y(:)]';

Ktest = kernel_Poly(X, X_grid,1);
prob = mlogistic(V_KLR*Ktest,c);

rate = zeros(size(prob,2),1);
for ii = 1:size(prob,2)
    rate(ii) = prob(1,ii) - prob(2,ii);
end

C_KLR = zeros(size(x));
for ii = 1:size(x,1)
    for jj = 1:size(x,2)
        C_KLR(jj,ii) = rate(jj + size(x,1)*(ii -1));
    end
end

%% Generating Test samples
theta = 30/360*2*pi;
R = [cos(theta) -sin(theta);sin(theta) cos(theta)];

X1_test = R*X1;
X1_test(1,:) = X1_test(1,:)*0.5;
X2_test = R*X2;
X2_test(2,:) = X2_test(2,:)*0.5;

X_test = [X1_test X2_test];

%% Importance Weighted Estimation
weight=KLIEP(X,X_test);

%% Importance Weighed KLR
V_IWKLR = klr_train(Ktrain,label,delta,itrNewton,weight);

Ktest = kernel_Poly(X, X_grid,1);
prob = mlogistic(V_IWKLR*Ktest,c);

rate = zeros(size(prob,2),1);
for ii = 1:size(prob,2)
    rate(ii) = prob(1,ii) - prob(2,ii);
end

C_IWKLR = zeros(size(x));
for ii = 1:size(x,1)
    for jj = 1:size(x,2)
        C_IWKLR(jj,ii) = rate(jj + size(x,1)*(ii -1));
    end
end

%% Plot Decision boundary
figure; 
contour(x,y,C_IWKLR,[0 0],'k'); hold on;
contour(x,y,C_KLR,[0 0], 'b');
plot(X1(1,:), X1(2,:), 'ro');
plot(X2(1,:), X2(2,:), 'x');
plot(X1_test(1,:), X1_test(2,:),'gv');
plot(X2_test(1,:), X2_test(2,:),'gs');
hold off;
legend('IWKLR', 'KLR', 'Class1-tr', 'Class2-tr','Class1-te', 'Class2-te');
axis([-6 6 -6 6]);

Ktest = kernel_Poly(X, X_test,1);
prob_IWKLR=mlogistic(V_IWKLR*Ktest,c);
prob_KLR=mlogistic(V_KLR*Ktest,c);
[val,index_KLR] = max(prob_KLR);
[val,index_IWKLR] = max(prob_IWKLR);

iden_rate_iwklr = 0;
iden_rate_klr = 0;
for ii = 1:size(prob_IWKLR,2)
    if index_IWKLR(ii) == label(ii)
        iden_rate_iwklr = iden_rate_iwklr +1;
    end
    
    if index_KLR(ii) == label(ii)
        iden_rate_klr = iden_rate_klr + 1;
    end
end


disp(sprintf('Identification rate of IWKLR: %g ', iden_rate_iwklr/size(prob_IWKLR,2)*100));
disp(sprintf('Identification rate of KLR  : %g ', iden_rate_klr/size(prob_KLR,2)*100));
