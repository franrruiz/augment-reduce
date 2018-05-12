#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <matrix.h>
#include <set>
#include "mex.h"
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#define input_N prhs[0]
#define input_D prhs[1]
#define input_K prhs[2]
#define input_X prhs[3]
#define input_Y prhs[4]
#define input_w prhs[5]
#define input_b prhs[6]
#define input_modelType prhs[7]
#define input_Nsamples prhs[8]
#define input_impSampStd prhs[9]
#define input_impSampMean prhs[10]
#define input_initSeed prhs[11]

#define output_llh plhs[0]
#define output_acc plhs[1]
#define output_llh_all plhs[2]

/* INPUTS:
 *
 *  0: N,            #datapoints
 *  1: D,            dimensionality
 *  2: K,            #classes
 *  3: X,            observation matrix (DxN); it *must* be a sparse matrix, and it *must* be transposed (i.e., DxN)
 *  4: Y,            labels (Nx1); must be 0-indexed
 *  5: w,            weight matrix (DxK)
 *  6: b,            biases (1xK)
 *  7: modelType,    1=softmax; 2=logistic; 3=probit
 *  8: Nsamples,     #samples for the MCMC approximation
 *  9: impSampStd,   std of the importance sampling distribution
 *  10: impSampMean, mean of the importance sampling distribution
 *  11: initSeed,    seed
 *
 */

/* OUTPUTS:
 *
 *  0: llh,          mean log-likelihood
 *  1: acc,          mean accuracy
 *  2: llh_all,      log-likelihood for each datapoint
 *
 */

inline double getdouble_2D(double *x, int n1, int n2, int N1, int N2);
inline void setdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);
inline void incrdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);
double logsumexp(double *p, int N);
double logsigmoid(double val);
double loggaussiancdf(double val);
inline double my_log(double val);
inline double my_pow2(double val);
double loggaussianpdf(double val, double mu, double sigma);
inline double my_exp(double val);
inline double my_log(double val);

const double myINFINITY=1.0e100;
const double LOG2PI = my_log(2.0*M_PI);

void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) {
    
    /**************************** Read inputs ****************************/
    int N = mxGetScalar(input_N);
    int D = mxGetScalar(input_D);
    int K = mxGetScalar(input_K);
    double *Xval = mxGetPr(input_X);
    mwIndex *Xir = mxGetIr(input_X);
    mwIndex *Xjc = mxGetJc(input_X);
    double *Y = mxGetPr(input_Y);
    double *weights = mxGetPr(input_w);
    double *biases = mxGetPr(input_b);
    int modelType = mxGetScalar(input_modelType);
    int Nsamples = mxGetScalar(input_Nsamples);
    double impSampStd = mxGetScalar(input_impSampStd);
    double impSampMean = mxGetScalar(input_impSampMean);
    int initSeed = mxGetScalar(input_initSeed);

    if(Xir==nullptr || Xjc==nullptr) {
        mexErrMsgTxt("The input matrix X must be sparse");
    }
    
    /******************** Allocate memory for output *********************/
    mwSize *dims = (mwSize*)calloc(2,sizeof(mwSize));
    dims[0] = N;
    dims[1] = 1;
    output_llh_all = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
    double *llh_all = mxGetPr(output_llh_all);
    free(dims);

    /************** Allocate memory for auxiliary variables **************/
    double *psi = new double[K];
    double *f_s = new double[Nsamples];

    /***************************** Main body *****************************/
    // set the seed
    // Set the seed
    gsl_rng *semilla = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(semilla, static_cast<unsigned long long>(initSeed));

    // loop over non-zero columns of the observation matrix
    int Yn;
    int Nelem_n;
    int Xjc_nn;
    int dd;
    double x_nd;
    double llh_mean = 0.0;
    double acc_mean = 0.0;
    double max_psi;
    int argmax;
    double u_s;
    for(int nn=0; nn<N; nn++) {
    	// get observed class
    	Yn = static_cast<int>(Y[nn]);

    	// compute psi
    	//   + set psi to bias
		for(int kk=0; kk<K; kk++) {
			psi[kk] = biases[kk];
		}
    	//   + loop over non-zero entries to accumulate inner products
    	Xjc_nn = Xjc[nn];
    	Nelem_n = Xjc[nn+1] - Xjc_nn;
    	for(int rr=0; rr<Nelem_n; rr++) {
    		dd = Xir[Xjc_nn+rr];
    		x_nd = Xval[Xjc_nn+rr];
    		for(int kk=0; kk<K; kk++) {
    			psi[kk] += x_nd * getdouble_2D(weights,dd,kk,D,K);
    		}
    	}

    	// compute log-lik
    	if(modelType==1) {
	    	// if softmax model
    		llh_all[nn] = psi[Yn] - logsumexp(psi, K);
    	} else if(modelType==2 || modelType==3) {
    		// if probit or logistic
    		for(int ss=0; ss<Nsamples; ss++) {
    			// sample from importance sampling proposal
    			u_s = impSampMean + impSampStd*gsl_ran_ugaussian(semilla);
    			// evaluate -logq
    			f_s[ss] = -loggaussianpdf(u_s, impSampMean, impSampStd);
    			// evaluate logPhi()
    			if(modelType==2) {
    				f_s[ss] += logsigmoid(u_s) + logsigmoid(-u_s);
    			} else if(modelType==3) {
    				f_s[ss] += loggaussianpdf(u_s, 0.0, 1.0);
    			}
    			// evaluate sum_logPhi()
    			for(int kk=0; kk<K; kk++) {
    				if(kk!=Yn) {
    					if(modelType==2) {
    						f_s[ss] += logsigmoid(u_s + psi[Yn] - psi[kk]);
    					} else if(modelType==3) {
    						f_s[ss] += loggaussiancdf(u_s + psi[Yn] - psi[kk]);
    					}
    				}
    			}
    		}
    		llh_all[nn] = logsumexp(f_s, Nsamples) - my_log(static_cast<double>(Nsamples));
    	} else {
    		mexErrMsgTxt("Unknown model type");
    	}
    	llh_mean += llh_all[nn];

    	// compute accuracy
    	max_psi = -myINFINITY;
    	argmax = -1;
    	for(int kk=0; kk<K; kk++) {
    		if(psi[kk]>max_psi) {
    			max_psi = psi[kk];
    			argmax = kk;
    		}
    	}
    	acc_mean += (Yn==argmax)?1.0:0.0;
    }
    llh_mean /= static_cast<double>(N);
    acc_mean /= static_cast<double>(N);
    
    /**************************** Return ****************************/
    output_llh = mxCreateDoubleScalar(llh_mean);
    output_acc = mxCreateDoubleScalar(acc_mean);
    gsl_rng_free(semilla);

    /**************************** Free memory ****************************/
    delete [] psi;
    delete [] f_s;
}


/************************ Auxiliary functions ************************/

inline double getdouble_2D(double *x, int n1, int n2, int N1, int N2) {
    return x[static_cast<unsigned long long>(N1)*n2+n1];
}

inline void setdouble_2D(double val, double *x, int n1, int n2, int N1, int N2) {
    x[static_cast<unsigned long long>(N1)*n2+n1] = val;
}

inline void incrdouble_2D(double val, double *x, int n1, int n2, int N1, int N2) {
    x[static_cast<unsigned long long>(N1)*n2+n1] += val;
}

double logsumexp(double *p, int N) {
	// find the maximum
	double maxVal = -myINFINITY;
	for(int nn=0; nn<N; nn++) {
		if(p[nn]>maxVal) {
			maxVal = p[nn];
		}
	}
	// accumulate the sum_exp
	double suma = 0.0;
	for(int nn=0; nn<N; nn++) {
		suma += my_exp(p[nn]-maxVal);
	}
	// return
	return(maxVal + my_log(suma));
}

double logsigmoid(double val) {
	double result;
	if(val>0.0) {
		result = -my_log(1.0+my_exp(-val));
	} else {
		result = val-my_log(1.0+my_exp(val));
	}
	return result;
}

double loggaussiancdf(double val) {
	return my_log(gsl_cdf_ugaussian_P(val));
}

inline double my_log(double val) {
    return (val<=0.0?-myINFINITY:gsl_sf_log(val));
}

inline double my_pow2(double val) {
	return val*val;
}

double loggaussianpdf(double val, double mu, double sigma) {
	double result;
	if(sigma==1.0) {
		result = - 0.5*LOG2PI - 0.5*my_pow2(val-mu);
	} else {
		result = - 0.5*LOG2PI - my_log(sigma) - 0.5*my_pow2(val-mu)/my_pow2(sigma);
	}
	return result;
}

inline double my_exp(double val) {
    return (val<-700.0?0.0:gsl_sf_exp(val));
}

/*************************************************************************/

