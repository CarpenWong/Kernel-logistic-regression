#include <stdio.h>

#include "mex.h"
#include "libklr.h"

void klr_train(double *V, double *Ktrain, int *label, Clibklr klr){
  klr.train(Ktrain, label);
  klr.get_V(V);
}

void return_V(double *V, mxArray *returned[]){
    returned[0] = mxCreateDoubleMatrix(1,1, mxREAL);
    V = mxGetPr(returned[0]);
}

void mexFunction( int Nreturned, mxArray *returned[], int Noperand, const mxArray *operand[] ){
  double *Ktrain;
  double *Ktrain_temp;
  double *label;
  int *label_temp;
  double *V;
  int n,n_test,m,c,wn,wm;
  double *delta;
  double *itrNewton;
  double *inweight;
  double *weight_temp;

  if(Noperand < 2){
      fprintf(stderr, "Error: \n");
      return;
  }

  Ktrain = mxGetPr(operand[0]);
  n      = mxGetN(operand[0]);
  m      = mxGetN(operand[0]);


  /* dummy variable */  
  V = (double*)malloc(sizeof(double));
  V[0] = 0;

  if( n != m ){
      printf("Error: Gram matrix should be symmetric.\n");
      return_V(V, returned);
      return;
  }
  
  if( n == 1 || m == 1){
      printf("Error: Gram matrix size should be greater than 2.\n");
       return_V(V, returned);
      return;
  }
  
  Ktrain_temp = (double*)malloc(sizeof(double)*n*m);
  for(int i = 0; i < m; i++){
      for(int j = 0; j < n; j++){
          Ktrain_temp[(i)+(j)*n] = Ktrain[(j) + i*n];
      }
  }
  
  label  = mxGetPr(operand[1]);
  n_test      = mxGetM(operand[1]);
  
  if(n != n_test){
       printf("Error: Numbers of Training and label samples are not same!\n"); 
       return_V(V, returned);
       return;
  }
  
  c = 0;
  label_temp = (int*)malloc(sizeof(int)*n);
  for(int i = 0; i < n_test; i++){
      if(c < label[i]){
          c = (int)label[i];
      }
      label_temp[i] = (int)label[i];
  }

  Clibklr klr(c,n);

  //KLR setting
  weight_temp = (double *)malloc(sizeof(double)*n);
  for(int i = 0; i < n; i++){
    weight_temp[i] = 1.0/(double)n;
  }

  klr.set_weight(weight_temp);
  klr.set_delta(0.01); 
  klr.set_itrCG(500);        
  klr.set_itrNewton(3);
  klr.set_itrKLR0(10);
  klr.set_tparam(0.001); 
 
  if(Noperand == 3){
      delta      = mxGetPr(operand[2]);
      klr.set_delta(delta[0]);
  }
  
  if(Noperand == 4){
      delta   = mxGetPr(operand[2]);
      itrNewton  = mxGetPr(operand[3]);
      klr.set_delta(delta[0]);
	  klr.set_itrNewton((int)itrNewton[0]);
  }
 
  if(Noperand >= 5){
      delta   = mxGetPr(operand[2]);
      itrNewton  = mxGetPr(operand[3]);
      inweight  = mxGetPr(operand[4]);
      wn      = mxGetN(operand[4]);
      wm      = mxGetM(operand[4]);
      
      if((wn == n_test && wm == 1) || (wn == 1 && wm == n_test)){    
          for(int i = 0; i < n; i++){
              weight_temp[i] = inweight[i];
          }
          klr.set_weight(weight_temp);
      }else{
          printf("Warning: Numbers of importance weight is wrong. We set weight = ones(%d, 1).\n",n); 
      }
      
      klr.set_delta(delta[0]); 
      klr.set_itrNewton((int)itrNewton[0]);
  }


  free(V); 
 
  returned[0] = mxCreateDoubleMatrix(c,n, mxREAL);
  V = mxGetPr(returned[0]); 
  klr_train(V, Ktrain_temp, label_temp, klr);
  free(Ktrain_temp);
  free(label_temp);
  free(weight_temp);
}
