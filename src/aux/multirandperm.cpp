#include "multirandperm.h"

#define input_N prhs[0]
#define input_K prhs[1]
#define input_ns prhs[2]
#define input_Y prhs[3]
#define input_flag_weights prhs[4]
#define input_weights prhs[5]
#define input_initSeed prhs[6]

#define output_samples plhs[0]

/* INPUTS:
 *
 *  0: N,            #datapoints
 *  1: K,            #classes
 *  2: ns,           #negative samples, must be <K
 *  3: Y,            true labels
 *  4: flag_weights  0 if no weights are provided, >0 if weighted
 *  5: weights       weights for each class (don't need to be normalized)
 *  6: initSeed,     seed
 *
 */

/* OUTPUTS:
 *
 *  0: samples,    random negative samples (N x ns)
 *
 */

void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) {
    
    /**************************** Read inputs ****************************/
    int N = mxGetScalar(input_N);
    int K = mxGetScalar(input_K);
    int ns = mxGetScalar(input_ns);
    int initSeed = mxGetScalar(input_initSeed);
    int flag_weights = mxGetScalar(input_flag_weights);
    double *weights = mxGetPr(input_weights);
    double *Y = mxGetPr(input_Y);
    
    /******************** Allocate memory for output *********************/
    mwSize *dims = (mwSize*)calloc(2,sizeof(mwSize));
    dims[0] = N;
    dims[1] = ns;
    output_samples = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
    double *samples = mxGetPr(output_samples);
    free(dims);
    
    /************** Allocate memory for auxiliary variables **************/
    double *vecK = (double*)calloc(K-1,sizeof(double));
    double *aux_ns = (double*)calloc(ns,sizeof(double));
    gsl_ran_discrete_t *weighted_sampler;
    if(flag_weights>0) {
	    weighted_sampler = gsl_ran_discrete_preproc(K, weights);
    }
    
    /***************************** Main body *****************************/
    // Set the seed
    gsl_rng *semilla = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(semilla,(unsigned long int)initSeed);
    
    // Fill in vecK
    for(int kk=0; kk<K-1; kk++) {
        vecK[kk] = kk+1;
    }
    
    // Main loop
    double val;
    if(static_cast<double>(ns)<static_cast<double>(K)/20.0 || flag_weights>0) {
	    for(int nn=0; nn<N; nn++) {
	    	std::set<int> curr_neg_samples;
	    	int ss = 0;
	    	while(ss<ns) {
	    		if(flag_weights==0){
		    		val = 1+gsl_rng_uniform_int(semilla, static_cast<unsigned long int>(K));
		    	} else {
		    		val = 1+gsl_ran_discrete(semilla, weighted_sampler);
		    	}
	    		std::set<int>::iterator iter = curr_neg_samples.find(val);
	    		if(val!=static_cast<int>(Y[nn]) && iter==curr_neg_samples.end()) {
	    			setdouble_2D(val, samples, nn, ss, N, ns);
	    			curr_neg_samples.insert(val);
		    		ss++;
	    		}
	    	}
    	}
    }
    else {
	    for(int nn=0; nn<N; nn++) {
	        gsl_ran_choose(semilla, aux_ns, ns, vecK, K-1, sizeof(double));
	        for(int ss=0; ss<ns; ss++) {
	            val = aux_ns[ss];
	            if(val>=Y[nn]) {
	                val += 1;
	            }
	            setdouble_2D(val, samples, nn, ss, N, ns);
	        }
	    }
	}
    
    /**************************** Free memory ****************************/
    gsl_rng_free(semilla);
    free(vecK);
    free(aux_ns);
    if(flag_weights>0) {
    	gsl_ran_discrete_free(weighted_sampler);
	}
}

/************************ Misc. Utility functions ************************/
/*
inline double my_exp(double val) {
    return (val<-700.0?0.0:gsl_sf_exp(val));
}

inline double my_log(double val) {
    return (val<=0?-myINFINITY:gsl_sf_log(val));
}
*/

/*************************** Get/Set functions ***************************/

inline double getdouble_2D(double *x, int n1, int n2, int N1, int N2) {
    return x[(unsigned long long)N1*n2+n1];
}

inline void setdouble_2D(double val, double *x, int n1, int n2, int N1, int N2) {
    x[(unsigned long long)N1*n2+n1] = val;
}

/*************************************************************************/

