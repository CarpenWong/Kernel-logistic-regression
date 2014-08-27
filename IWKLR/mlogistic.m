function pr = mlogistic(f,K)
f=f-ones(K,1)*max(f);
pr=exp(f)./(ones(K,1)*sum(exp(f)));

 