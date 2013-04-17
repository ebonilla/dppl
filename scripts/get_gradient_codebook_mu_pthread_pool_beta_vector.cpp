#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include <limits>
#include <algorithm>
#include <pthread.h>
/*
 * 09/08/2011: Modified to implement a pool of threads we do not want 
 * to create too many thread. 
 * Most of the thread-pool handling code is taking from 
 * http://users.actcom.co.il/~choo/lupg/tutorials/multi-thread/multi-thread.html
 *
 * 05/10/2011 Modified to accept beta as a vector
 */

/* macros for debugging */
#ifdef DEBUG

#include "mex.h"
#define MPRINT_MSG(_msg)  mexPrintf("[%s|%s, %3d]: %s\n", __FILE__, __func__, __LINE__, _msg);
#define MPRINT_INT(_val)  mexPrintf("[%s|%s, %3d]: %s = %d\n", __FILE__, __func__, __LINE__, #_val, _val);
#define MPRINT_DBL(_val)  mexPrintf("[%s|%s, %3d]: %s = %f\n", __FILE__, __func__, __LINE__, #_val, _val);
#define MPRINT_PTR(_val)  mexPrintf("[%s|%s, %3d]: %s = %p\n", __FILE__, __func__, __LINE__, #_val, _val);

#else       /* #ifdef DEBUG */

/* do not print messages and values */
#define MPRINT_MSG(_msg)
#define MPRINT_INT(_val)
#define MPRINT_DBL(_val)
#define MPRINT_PTR(_val)

/* assert needs that when no debugging is desired and therefore, assertion is ignored */
#ifndef NDEBUG
#define NDEBUG 1
#endif

#endif       /* #ifdef DEBUG */
 


/* Number of threads to handle requests (each request is to deal with a data-point) */
#define MAX_NTHREADS  12
#define PP(i,j) PP[j*K+i]



double *grad_mu;
double *all_x, *Mu, *y, *P, *W;
double *beta_x;

struct thread_data{
int N, K, D_x,  L_mu, N_all, C; 
int c_class, n;
int low, up; 
double *X, *G;
};


/* We declare this globally so all the threads have access to the data 
 * We will create as many entries as requests (data-points)
 */
struct thread_data* td_array;
 
/* Other thread data structures here */
pthread_mutex_t request_mutex ; /* global mutex for our program. assignment initializes it. */

// pthread_cond_t  got_request = PTHREAD_COND_INITIALIZER; /* global condition variable for our program. assignment initializes it. */
pthread_cond_t  got_request; /* dynamically initialized in main */

int num_requests;	/* number of pending requests, initially none */
int done_creating_requests;	/* are we done creating new requests? */
struct request { /* format of a single request. */
    int number;		    /* number of the request                  */
    struct request* next;   /* pointer to next request, NULL if none. */
};
struct request* requests = NULL;     /* head of linked list of requests. */
struct request* last_request = NULL; /* pointer to last request.         */

pthread_mutex_t mutexsum; /* to accumulate the sum of gradients */



using namespace std;

/*
    prhs[0]: all_x: data matrix of size up(end) x D   
 *  prhs[1]: low
 *  prhs[2]: up
 *  prhs[3]: Mu 
 *  prhs[4]: P: 
 *  prhs[5]: W:
 *  prhs[6]: N 
 *  prhs[7]: D_x
 *  prhs[8]: K 
 *  prhs[9]: beta_x
 *  prhs[10]: y 
 *  prhs[10]: C
 *  plhs[0]: grad_mu: 
 *
 */



// a: K x D
// b: N_x x D
// G: K x N_x
void sq_dist(double *a, double *b, int K, int N_x, int D, double *G){
    double  z;
    
    for (int kk=0; kk < K*N_x; kk++) G[kk] = 0.0;
    
    for (int k=0; k<K; k++){ 
        for (int n=0; n<N_x; n++) {
            
            for (int d=0; d<D; d++) {  
                z = a[K*d + k] - b[N_x*d + n]; 
                
                G[K*n + k]  = G[K*n + k] + z*z;
            }
        
        }
    }

}


// PP: K x N_x
void get_softmax_generic(int K, int N_x, double *PP){
   int i, j;
   double realmax, realmin, mincut, maxcut;  
   double Z[N_x];
     
   /*
    * CHANGED ON 07/09/2011 to control overflow differently
    * 
   realmax = std::numeric_limits<double>::max();
   realmin = std::numeric_limits<double>::min();
   maxcut = log(realmax) - log(K);
   mincut = log(realmin);
    
   for (j = 0; j < N_x; j++){
        Z[j] = 0;
        for ( i = 0; i < K; i++){ 
            PP(i,j) = - beta_x *  PP(i,j);
            PP(i,j) =   std::min(PP(i,j), maxcut);
            PP(i,j) =   std::max(PP(i,j), mincut); 
            PP(i,j)   = exp(PP(i,j)); 
            
            Z[j] = Z[j] + PP(i,j);
            
        }
   }   
   for (j = 0; j < N_x; j++){
            for ( i = 0; i < K; i++){ 
                PP(i,j) = PP(i,j)/Z[j];
                if (PP(i,j) < realmin) 
                    PP(i,j) = realmin;
            }
  }
  */
   double maxi[N_x];
    for (j = 0; j < N_x; j++){ /* determines maximum for each colum */
        i = 0; PP(i,j)  = - beta_x[i] *  PP(i,j);
        maxi[j] = PP(i,j);
        for (i=1; i < K; i++){
            PP(i,j) = - beta_x[i] *  PP(i,j); /* updates P(i,j) before using it */
            maxi[j] = std::max(PP(i,j), maxi[j]); 
        }
    } 
    
    /* Subtracts the max, exponentiates and computes normalization constant */
    for (j = 0; j < N_x; j++){ /* determines maximum for each colum */
        Z[j] = 0;   
        for (i=0; i < K; i++){
            PP(i,j)  = PP(i,j) - maxi[j]; /* updates P(i,j) before using it */
            PP(i,j) = exp(PP(i,j));
            Z[j] = Z[j] + PP(i,j);  
        }
        
        /*printf("Z[%d]=%.3f\n", j, Z[j]);*/
        
    }
      
    
     /* Normalization here */
     for (j = 0; j < N_x; j++)
            for ( i = 0; i < K; i++) 
                PP(i,j) = PP(i,j)/Z[j];
        
    
}  


void get_current_x(int N_all, int D_x, int low, int up, double *X){
   
    int N_x = up - low + 1;
    int row;
    
    
    
    
      for (int i=0; i < N_x; i++){
        
       row = low + i;
        for (int j=0; j < D_x; j++){            
             X[N_x*j+i] = all_x[N_all*j+row];
        }
       
        
    }
     
     
     
     
}
 
 
void upgrade_gradient_single_point(double *X, double *G, int N, int K, int D_x, int L_mu, int N_all, int C,
                                    int c_class, int low, int up, int n){
            
 
            double dz_dmu;
            double val[L_mu];            
            double local_grad_mu[L_mu];
            int ptr;                    
            int N_x = up - low + 1;                        
            
                        
            get_current_x(N_all, D_x, low, up, X); // gets corresponding slice of X                                         
            sq_dist(Mu, X, K, N_x, D_x, G);
            get_softmax_generic(K, N_x, G);
            

            
            
            
            for (int ll=0; ll< L_mu; ll++) local_grad_mu[ll] = 0;
            
            for (int k = 0; k < K; k++){      // for each z_k there is a different gradient 
                
                for (int ll=0; ll< L_mu; ll++) val[ll] = 0;   // cleaning up val
                
                
                
                // dLn_dznk here
                double dL_dz =  W[C*k + c_class-1];                 
                for (int c = 0; c < C; c++){
                        dL_dz = dL_dz -  P[N*c + n]*W[C*k + c]; 
                }
                
                
                
                for (int l = 0 ; l < K; l++){                                       
                                 
                    for (int m = 0; m < N_x; m++){  // sum over all the x in S_m                        
                                    
                        for (int d=0; d < D_x; d++){                  
                            ptr = l*D_x + d;        
                           if (l==k){
                            val[ptr] = val[ptr] + 
                                        beta_x[k]*G[m*K+k]*(1-G[m*K+k])*( X[d*N_x+m] - Mu[d*K +k] );
                           }else{
                            val[ptr] = val[ptr] + 
                                       beta_x[l]*G[m*K+k]*G[m*K+l]*( Mu[d*K +l] -X[d*N_x+m]);
                           }
                        }   // d         
                    }      // m               
                }  // l
                
                
                for (int ll=0; ll< L_mu; ll++){ 
                    dz_dmu = 2*val[ll]; 
                    
                    // Added on 28/05/2011
                    dz_dmu  = dz_dmu/N_x;
                    
                    
                    local_grad_mu[ll] = local_grad_mu[ll] + dL_dz*dz_dmu;

                }
                
               
                
            } // k
            
            // Update global grad_mu
            pthread_mutex_lock(&mutexsum);
            for (int ll=0; ll< L_mu; ll++){ 
                    grad_mu[ll] = grad_mu[ll] + local_grad_mu[ll];
                    
            }  
            pthread_mutex_unlock(&mutexsum);        
                     
}


/********************* Thread handling code from here ********************/

/*
 * function add_request(): add a request to the requests list
 * algorithm: creates a request structure, adds to the list, and
 *            increases number of pending requests by one.
 * input:     request number, linked list mutex.
 * output:    none.
 */
void add_request(int request_num, pthread_mutex_t* p_mutex, pthread_cond_t*  p_cond_var){
    int rc;	                    /* return code of pthreads functions.  */
    struct request* a_request;      /* pointer to newly added request.     */

    /* create structure with new request */
    a_request = (struct request*) mxMalloc(sizeof(struct request));
    if (!a_request) { /* malloc failed?? */
        fprintf(stderr, "add_request: out of memory\n");
        exit(1);
    }
    a_request->number = request_num;
    a_request->next = NULL;

    /* lock the mutex, to assure exclusive access to the list */
    rc = pthread_mutex_lock(p_mutex);

    /* add new request to the end of the list, updating list */
    /* pointers as required */
    if (num_requests == 0) { /* special case - list is empty */
	requests = a_request;
	last_request = a_request;
    }
    else {
	last_request->next = a_request;
	last_request = a_request;
    }

    /* increase total number of pending requests by one. */
    num_requests++;

#ifdef DEBUG    
    mexPrintf("add_request: added request with id '%d'\n", a_request->number);
#endif /* DEBUG */


    /* unlock mutex */
    rc = pthread_mutex_unlock(p_mutex);

    /* signal the condition variable - there's a new request to handle */
    rc = pthread_cond_signal(p_cond_var);
    
}

 
/*
 * function get_request(): gets the first pending request from the requests list
 *                         removing it from the list.
 * algorithm: creates a request structure, adds to the list, and
 *            increases number of pending requests by one.
 * input:     request number, linked list mutex.
 * output:    pointer to the removed request, or NULL if none.
 * memory:    the returned request need to be freed by the caller.
 */
struct request* get_request(pthread_mutex_t* p_mutex){
    int rc;	                    /* return code of pthreads functions.  */
    struct request* a_request;      /* pointer to request.                 */

    /* lock the mutex, to assure exclusive access to the list */
    rc = pthread_mutex_lock(p_mutex);

    if (num_requests > 0) {
	a_request = requests;
	requests = a_request->next;
	if (requests == NULL) { /* this was the last request on the list */
	    last_request = NULL;
	}
	/* decrease the total number of pending requests */
	num_requests--;
    }
    else { /* requests list is empty */
	a_request = NULL;
    }

    /* unlock mutex */
    rc = pthread_mutex_unlock(p_mutex);

    /* return the request to the caller. */
    return a_request;
}

/*
 * function handle_request(): handle a single given request.
 * algorithm: prints a message stating that the given thread handled
 *            the given request.
 * input:     request pointer, id of calling thread.
 * output:    none.
 */
void
handle_request(struct request* a_request, int thread_id)
{
    if (a_request) {
     int n = a_request->number;
     
     upgrade_gradient_single_point(td_array[n].X, td_array[n].G, td_array[n].N, 
                td_array[n].K, td_array[n].D_x, td_array[n].L_mu, 
                    td_array[n].N_all, td_array[n].C, td_array[n].c_class, 
             td_array[n].low, td_array[n].up, td_array[n].n);
          
#ifdef DEBUG
     mexPrintf("Thread '%d' handled request '%d'\n", thread_id, a_request->number);
#endif /* DEBUG */

    }
}



/*
 * function handle_requests_loop(): infinite loop of requests handling
 * algorithm: forever, if there are requests to handle, take the first
 *            and handle it. Then wait on the given condition variable,
 *            and when it is signaled, re-do the loop.
 *            increases number of pending requests by one.
 * input:     id of thread, for printing purposes.
 * output:    none.
 */
void *handle_requests_loop(void* data){
    int rc;	                    /* return code of pthreads functions.  */
    struct request* a_request;      /* pointer to a request.               */
    int thread_id = *((int*)data);  /* thread identifying number           */

    /* lock the mutex, to access the requests list exclusively. */
    rc = pthread_mutex_lock(&request_mutex);


    /* do forever.... */
    while (1) {
	if (num_requests > 0) { /* a request is pending */
	    a_request = get_request(&request_mutex);
	    if (a_request) { /* got a request - handle it and free it */
            /* unlock mutex - so other threads would be able to handle */
            /* other reqeusts waiting in the queue paralelly.          */
            rc = pthread_mutex_unlock(&request_mutex);
            handle_request(a_request, thread_id);
            
            // mxFree(a_request);
            
            
            rc = pthread_mutex_lock(&request_mutex);  /* and lock the mutex again. */
	    }
	}
	else {
	    /* the thread checks the flag before waiting */
	    /* on the condition variable.                           */
	    /* if no new requests are going to be generated, exit.  */
	    if (done_creating_requests) {
            pthread_mutex_unlock(&request_mutex);

#ifdef DEBUG            
            mexPrintf("thread '%d' exiting\n", thread_id);  
#endif /* DEBUG */
            pthread_exit(NULL);
	    }
	    else {
#ifdef DEBUG            
    		mexPrintf("thread '%d' going to sleep\n", thread_id);
#endif /* DEBUG */
	    }

	    /* wait for a request to arrive. note the mutex will be */
	    /* unlocked here, thus allowing other threads access to */
	    /* requests list.                                       */
	    rc = pthread_cond_wait(&got_request, &request_mutex);
	    /* and after we return from pthread_cond_wait, the mutex  */
	    /* is locked again, so we don't need to lock it ourselves */
	}
  }
}




/*
 * This was used before. Now, everything is called by handle_requests_loop which calls handle_request
void *call_update_gradient(void *threadarg){
    struct thread_data *td = (struct thread_data *) threadarg;
    
    upgrade_gradient_single_point(td->X, td->G, td->N, td->K, td->D_x, td->L_mu, 
                    td->N_all, td->C, td->c_class, td->low, td->up, td->n);
      
    
   pthread_exit(NULL);   

}
 */



/* IMPORTANT: When calling this function low and high must be already in 
 * C convention: 0, ..., length(v)-1 */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{    
    /* N is the numner of data-points (requests), which, in general is
     considerably higher than the number of threads */
    int NTHREADS, N,  K,  D_x,  L_mu,  N_all,  C, n, t; 
    double *low, *up;
    void *status;
    int rc;    
    
    struct timespec delay;			 /* used for wasting time */
  
                   
    all_x  = mxGetPr(prhs[0]); N_all  = mxGetM(prhs[0]);
    low    = mxGetPr(prhs[1]);
    up     = mxGetPr(prhs[2]);
    Mu     = mxGetPr(prhs[3]);
    P      = mxGetPr(prhs[4]);
    W      = mxGetPr(prhs[5]);
    N      = *mxGetPr(prhs[6]);
    D_x    = *mxGetPr(prhs[7]);
    K      = *mxGetPr(prhs[8]);
    beta_x = mxGetPr(prhs[9]);  
    y      = mxGetPr(prhs[10]);  
    C      = *mxGetPr(prhs[11]); // Number of classes  
    L_mu  = K*D_x; /* size of parameter vector and gradient */                
    plhs[0]    = mxCreateDoubleMatrix(1, L_mu, mxREAL);    
    grad_mu    = mxGetPr(plhs[0]); 
    
    
    NTHREADS = MAX_NTHREADS;
    if (N < NTHREADS) NTHREADS = N;
    
    
    
    num_requests = 0;	/* number of pending requests, initially none */
    done_creating_requests = 0;	/* are we done creating new requests? */
        
    /* Intialization of mutex here */
    pthread_mutexattr_t mutex_attr;
    rc = pthread_mutexattr_init(&mutex_attr); if(rc != 0){ mexPrintf("error 1\n"); exit(1); }
    rc = pthread_mutexattr_settype(&mutex_attr,PTHREAD_MUTEX_RECURSIVE); if(rc != 0){ mexPrintf("error 2\n"); exit(2); }
    rc = pthread_mutex_init(&request_mutex, &mutex_attr); if(rc != 0){ mexPrintf("error 3\n"); exit(3); }
    
    /* initialization of conditiona variable */
    rc = pthread_cond_init(&got_request, NULL); if(rc != 0){ mexPrintf("error initializing got_request\n"); exit(1); }
    
    /* this contains the data for all the data-points */
    td_array = (struct thread_data *) mxMalloc(N*sizeof(struct thread_data));
      
    
    pthread_t threads[NTHREADS];
    int thr_id[NTHREADS]; /* thread IDs */
    pthread_attr_t attr;
    pthread_mutex_init(&mutexsum, NULL);

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    
    /* Create all the threads to handle the requests */
    for (t=0; t<NTHREADS; t++){
        thr_id[t] = t;
        rc = pthread_create(&threads[t], &attr, handle_requests_loop, (void*)&thr_id[t]);        
        
        if (rc){
         mexPrintf("ERROR; return code from pthread_create() is %d\n", rc);
         return;
        }                 
    }
        
   /* We store all the data in td_array */
   for (n = 0; n < N; n++){                
   
       td_array[n].N       = N;
       td_array[n].K       = K;
       td_array[n].D_x     = D_x;
       td_array[n].L_mu    = L_mu;
       td_array[n].N_all   = N_all;
       td_array[n].C       = C;
       td_array[n].c_class = y[n];
       td_array[n].low     = low[n];
       td_array[n].up      = up[n];
       td_array[n].n       = n;
       
       int N_x = td_array[n].up - td_array[n].low + 1;
       td_array[n].X       = (double *) mxMalloc(N_x * td_array[n].D_x * sizeof(double));        
       td_array[n].G       = (double *) mxMalloc(N_x * td_array[n].K * sizeof(double)); 
       
       /* adds requests and pauses for  bit to allow other threads to handle
        * some requestst 
        */      
       	add_request(n, &request_mutex, &got_request);

       	if (rand() > 3*(RAND_MAX/4)) { /* this is done about 25% of the time */
            delay.tv_sec = 0;
            delay.tv_nsec = 1;
            nanosleep(&delay, NULL);
        }
       
   }   
   
   
   /* the main thread modifies the flag     
      * to tell its handler threads no new requests will 
      * be generated.                                    
      * notify our threads we're done creating requests. 
   */
   rc = pthread_mutex_lock(&request_mutex);
   done_creating_requests = 1;
   rc = pthread_cond_broadcast(&got_request);
   rc = pthread_mutex_unlock(&request_mutex);
        
 
   for(t=0; t < NTHREADS; t++) {
      rc = pthread_join(threads[t], &status);
      if (rc) {
         mexPrintf("ERROR; return code from pthread_join()  is %d\n", rc);
         exit(-1);
         }
   }
  
   /*
   for (n = 0; n < N; n++){
      mxFree(td_array[n].X);
      mxFree(td_array[n].G);     
   }
   mxFree(td_array);
    
  pthread_attr_destroy(&attr);
  pthread_mutex_destroy(&mutexsum); 
  */
  
  /*mexPrintf("Main: All jobs completed. Exiting.\n");*/
 
  

} 
     


