%  Multi-class Importance Weighted Kernel Logistic Regression
%    
%   USAGE:     V = klr_train(Ktrain,label,delta,itrNewton,weight);
%
%  INPUTS:
%       Ktrain:     Gram matrix of training data.
%        label:     label information eg. label = [1, ... 1, 2, ...2, ... , N, .... N]
%        delta:     Regularization parameter
%    itrNewton:     Number of iterations for Newton Method.
%       weight:     Importance weight of each training datasamples.
% 
%  OUTPUTS:
%            V:     Estimated discriminating function
% 
%  Examples: 
%            case 1. Kernel Logistic Regression
%               
%               delta      = 0.00001; 
%               itrNewton  = 5;
%               
%               V_KLR = klr_train(Ktrain,label,delta, itrNewton);
%               prob = mlogistic(V_KLR*Ktest,c); % c: Number of classes.
%
%            case 2. Importance Weighted Kernel Logistic Regression
%              
%               delta      = 0.00001; 
%               itrNewton  = 5; 
%               
%               weight = KLIEP(X_train,X_test);
%               V_IWKLR  = klr_train(Ktrain,label,delta, itrNewton, weight);
%               prob = mlogistic(V_IWKLR*Ktest,c); % c: Number of classes.
%
%        (c) Makoto Yamada, Department of Compter Science, Tokyo Institute of Technology, Japan.
%            yamada@sg.cs.titech.ac.jp, 