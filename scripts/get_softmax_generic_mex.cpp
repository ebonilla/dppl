#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include <limits>
#include <algorithm>

#define P(i,j) P_[j*K+i]

using namespace std;

/*
    prhs[0]: Mu: KxD matrix
 *  prhs[1]: x: NXD data matrix
 *  prhs[2]: beta_x
 *
 * plhs[0]: P: KxN matrix: Each column of P is normalized
 *
 */

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    int N, K, D, i, j;
    double realmax, realmin, mincut, maxcut;
    
    double *Mu;
    double *x;
    double beta_x;
    
    double *P_;
    
    
    mxArray *rhs[2];
    
             
    N = mxGetM(prhs[1]);
    D = mxGetN(prhs[1]);
    K = mxGetM(prhs[0]);
    beta_x = *mxGetPr(prhs[2]);  

    double Z[N];
    plhs[0]     = mxCreateDoubleMatrix(K, N, mxREAL);
    rhs[0]      = mxCreateDoubleMatrix(D, K,  mxREAL);
    rhs[1]      =  mxCreateDoubleMatrix(D, N,  mxREAL);

    /* sq_dist(mu', x') */
    mexCallMATLAB(1, &rhs[0], 1, (mxArray **) &prhs[0], "transpose");
    mexCallMATLAB(1, &rhs[1], 1,  (mxArray **) &prhs[1], "transpose");
    mexCallMATLAB(1, plhs, 2,  rhs, "sq_dist");

    
    P_    =  mxGetPr(plhs[0]); 
    
    /*for (j = 0; j < N; j++)
        for (i = 0; i < K; i++)
            mexPrintf("P(%d,%d)=%f\n", i,j,P(i,j));
     */

    /*
     * COMMENTED OUT ON 07/09/2011 AS NOW WITH CONTROL OVERFLOW DIFFERENTLY
     */
    /*
    realmax = std::numeric_limits<double>::max();
    realmin = std::numeric_limits<double>::min();
    maxcut = log(realmax) - log(K);
    mincut = log(realmin);
    
    for (j = 0; j < N; j++){
        Z[j] = 0;
        for ( i = 0; i < K; i++){ 
            P(i,j) = - beta_x *  P(i,j);
            P(i,j) =   std::min(P(i,j), maxcut);
            P(i,j) =   std::max(P(i,j), mincut); 
            P(i,j)   = exp(P(i,j));             
            Z[j] = Z[j] + P(i,j);            
        }
    } 
     *
     */   
    /* Normalization here */
    /*
     for (j = 0; j < N; j++){
            for ( i = 0; i < K; i++){ 
                P(i,j) = P(i,j)/Z[j];
                if (P(i,j) < realmin) 
                    P(i,j) = realmin;
            }
     }*/
    
    double maxi[N];
    for (j = 0; j < N; j++){ /* determines maximum for each colum */
        i = 0; P(i,j)  = - beta_x *  P(i,j);
        maxi[j] = P(i,j);
        for (i=1; i < K; i++){
            P(i,j) = - beta_x *  P(i,j); /* updates P(i,j) before using it */
            maxi[j] = std::max(P(i,j), maxi[j]); 
            
           // mexPrintf("P(%d,%d)=%f --> maxi[%d]=%f\n",i, j, P(i,j), j,maxi[j]);

        }
    } 
    
    /* Subtracts the max, exponentiates and computes normalization constant */
    for (j = 0; j < N; j++){ /* determines maximum for each colum */
        Z[j] = 0;   
        for (i=0; i < K; i++){
            P(i,j)  = P(i,j) - maxi[j]; /* updates P(i,j) before using it */
            
            //mexPrintf("P(%d,%d)=%f-->", i,j,P(i,j));

                        
            P(i,j) = exp(P(i,j));
            Z[j] = Z[j] + P(i,j);  
            
            //mexPrintf("P(%d,%d)=%f\n", i,j,P(i,j));
    
        }
        
    }
    
    
     /* Normalization here */
     for (j = 0; j < N; j++)
            for ( i = 0; i < K; i++) 
                P(i,j) = P(i,j)/Z[j];
  

    
    
    
    
    
      
    
}
 

