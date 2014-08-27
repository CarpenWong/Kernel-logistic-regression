%% Kernel Logistic Regression
clear all;
close all;

%Number of classes
c = 2;     

n1 = 500; 
n2 = 500;
n  = n1 + n2;
%Training data
X1 = randn(2,n1) - [3;-0.5]*ones(1,n1);
X2 = randn(2,n2) + [1;-0.5]*ones(1,n2);
 
X = [X1 X2];

%Label Data
label = [ones(n1,1); 2*ones(n2,1)];

%KLR parameter
delta   = 0.000001;    
itrNewton = 5;
tparam  = 0.000001;   

%Cross Validation
[ind1,bb] = find(label == 1);
[ind2,bb] = find(label == 2);

isCV = 1;
optdeg = 1;
optdelta = delta;
if isCV == 1

    optdelta = 0.1;
    maxCVscore = -1;

    cv_index1 = randperm(n1);
    cv_index2 = randperm(n2);
   
    fold=5;
    cv_split1 = floor([0:n1-1]*fold./n1)+1;
    cv_split2 = floor([0:n2-1]*fold./n2)+1;

    kCVscoreall  = zeros(1,6);
                    
    for ii = 1:1
        deg = ii;
        Ktrain = kernel_Poly(X,X,deg);
        for jj = 1:6
            delta = 10^(-(jj));
            kCVscore = 0.0;
            
            for kk=1:fold
                cv_train_index=[ind1(cv_index1(cv_split1~=kk)); ind2(cv_index2(cv_split2~=kk))];
                cv_test_index= [ind1(cv_index1(cv_split1==kk)); ind2(cv_index2(cv_split2==kk))];
           
                V_KLR = klr_train(Ktrain(cv_train_index,cv_train_index),label(cv_train_index),delta,itrNewton);
                prob = mlogistic(V_KLR*Ktrain(cv_train_index,cv_test_index),c);
             
                [val, mindex] = max(prob);
                temp = label(cv_test_index) - mindex'; 
                [val, num] = find(temp == 0);
                rate = sum(num)/length(cv_test_index);
                kCVscore = kCVscore + rate/fold;
            end

            if kCVscore > maxCVscore
                optdeg   = deg;
                optdelta = delta;
               maxCVscore = kCVscore;
            end
            kCVscoreall(ii,jj) = kCVscoreall(ii,jj) + kCVscore;
        end
    end
end

%Kernel Computation
Ktrain = kernel_Poly(X,X,optdeg);
%KLR training 
tic
V_KLR = klr_train(Ktrain,label,optdelta,itrNewton);
toc

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

%% Plot Decision boundary
figure; 
contour(x,y,C_KLR,[0 0],'k'); hold on;
plot(X1(1,:), X1(2,:), 'ro');
plot(X2(1,:), X2(2,:), 'x');
hold off;
legend('KLR', 'Class1-tr', 'Class2-tr');
axis([-6 6 -6 6]);
