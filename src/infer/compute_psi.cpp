#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <matrix.h>
#include <set>
#include "mex.h"

#define input_N prhs[0]
#define input_D prhs[1]
#define input_K prhs[2]
#define input_ns prhs[3]
#define input_X prhs[4]
#define input_Y prhs[5]
#define input_negsamples prhs[6]
#define input_w prhs[7]
#define input_b prhs[8]

#define output_Psi_obs plhs[0]
#define output_Psi_ns plhs[1]

/* INPUTS:
 *
 *  0: N,            #datapoints
 *  1: D,            dimensionality
 *  2: K,            #classes
 *  3: ns,           #negative samples, must be <K
 *  4: X,            observation matrix (NxD); it *must* be a sparse matrix
 *  5: Y,            labels (Nx1); must be 0-indexed
 *  6: negsamples,   indices to negative samples (Nxns); must be 0-indexed
 *  7: w,            weight matrix (DxK)
 *  8: b,            biases (1xK)
 *
 */

/* OUTPUTS:
 *
 *  0: Psi_obs,      inner products w*x+b for the observed values (N x 1)
 *  1: Psi_ns,       inner products w*x+b for the negative samples (N x ns)
 *
 */

inline double getdouble_2D(double *x, int n1, int n2, int N1, int N2);
inline void setdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);
inline void incrdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);

void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) {
    
    /**************************** Read inputs ****************************/
    int N = mxGetScalar(input_N);
    int D = mxGetScalar(input_D);
    int K = mxGetScalar(input_K);
    int ns = mxGetScalar(input_ns);
    double *Xval = mxGetPr(input_X);
    mwIndex *Xir = mxGetIr(input_X);
    mwIndex *Xjc = mxGetJc(input_X);
    double *Y = mxGetPr(input_Y);
    double *negsamples = mxGetPr(input_negsamples);
    double *weights = mxGetPr(input_w);
    double *biases = mxGetPr(input_b);
    
    if(Xir==nullptr || Xjc==nullptr) {
        mexErrMsgTxt("The input matrix X must be sparse");
    }
    
    /******************** Allocate memory for output *********************/
    mwSize *dims = (mwSize*)calloc(2,sizeof(mwSize));
    dims[0] = N;
    dims[1] = 1;
    output_Psi_obs = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
    double *Psi_obs = mxGetPr(output_Psi_obs);
    dims[0] = N;
    dims[1] = ns;
    output_Psi_ns = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
    double *Psi_ns = mxGetPr(output_Psi_ns);
    free(dims);
    
    /************** Allocate memory for auxiliary variables **************/

    /***************************** Main body *****************************/

    int Yn;

    // initialize the output matrices to the bias term
    for(int nn=0; nn<N; nn++) {
        Yn = static_cast<int>(Y[nn]);
        Psi_obs[nn] = biases[Yn];
        for(int ss=0; ss<ns; ss++) {
            Yn = static_cast<int>(getdouble_2D(negsamples, nn, ss, N, ns));
            setdouble_2D(biases[Yn], Psi_ns, nn, ss, N, ns);
        }
    }

    // loop over non-zero columns of the observation matrix
    int Nelem_d;
    int nn;
    double val;
    double result;
    int Xjc_dd;
    for(int dd=0; dd<D; dd++) {
        Xjc_dd = Xjc[dd];
        Nelem_d = Xjc[dd+1] - Xjc_dd;
        // loop over non-zero rows within that column
        for(int rr=0; rr<Nelem_d; rr++) {
            nn = Xir[Xjc_dd+rr];
            Yn = static_cast<int>(Y[nn]);
            val = Xval[Xjc_dd+rr];
            result = val * getdouble_2D(weights, dd, Yn, D, K);
            Psi_obs[nn] += result;
            // loop over negative samples
            for(int ss=0; ss<ns; ss++) {
                Yn = static_cast<int>(getdouble_2D(negsamples, nn, ss, N, ns));
                result = val * getdouble_2D(weights, dd, Yn, D, K);
                incrdouble_2D(result, Psi_ns, nn, ss, N, ns);
            }
        }
    }
    
    /**************************** Free memory ****************************/
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
    return x[static_cast<unsigned long long>(N1)*n2+n1];
}

inline void setdouble_2D(double val, double *x, int n1, int n2, int N1, int N2) {
    x[static_cast<unsigned long long>(N1)*n2+n1] = val;
}

inline void incrdouble_2D(double val, double *x, int n1, int n2, int N1, int N2) {
    x[static_cast<unsigned long long>(N1)*n2+n1] += val;
}

/*************************************************************************/

