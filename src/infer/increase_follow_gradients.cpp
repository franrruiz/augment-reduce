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
#define input_w_Gt prhs[9]
#define input_b_Gt prhs[10]
#define input_dLdpsi prhs[11]
#define input_dLdpsi_botev prhs[12]
#define input_sviFactor prhs[13]
#define input_s2w prhs[14]
#define input_s2b prhs[15]
#define input_step_eta prhs[16]
#define input_step_kappa prhs[17]
#define input_step_T prhs[18]
#define input_step_it prhs[19]

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
 *  9: w_Gt,         Gt matrix for the stepsize schedule for the weights (DxK)
 *  10: b_Gt,        Gt matrix for the stepsize schedule for the biases (1xK)
 *  11: dLdpsi,      Gradient of the ELBO wrt to the parameters psi (Nxns)
 *  12: dLdpsi_botev, Gradient of the ELBO wrt to the parameters psi corresponding to the positive classes (Nx1) [only for 'botev' method]
 *  13: sviFactor,   SVI factor to rescale the gradients
 *  14: s2w,         prior variance on the weights
 *  15: s2b,         prior variance on the biases
 *  16: step_eta,    parameter for 'stan' stepsize schedule: learning rate
 *  17: step_kappa,  parameter for 'stan' stepsize schedule: exponent
 *  18: step_T,      parameter for 'stan' stepsize schedule: delay
 *  19: step_it,     parameter for 'stan' stepsize schedule: iteration
 *
 */

/* OUTPUTS:
 *
 * (w, b, Gt_w, and Gt_b updated by reference)
 *
 */

inline double getdouble_2D(double *x, int n1, int n2, int N1, int N2);
inline void setdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);
inline void incrdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);
inline double my_pow2(double val);

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
    double *Gt_w = mxGetPr(input_w_Gt);
    double *Gt_b = mxGetPr(input_b_Gt);
    double *dLdpsi = mxGetPr(input_dLdpsi);
    double *dLdpsi_botev = nullptr;
    if(!mxIsEmpty(input_dLdpsi_botev)) {
        dLdpsi_botev = mxGetPr(input_dLdpsi_botev);
    }
    double sviFactor = mxGetScalar(input_sviFactor);
    double s2w = mxGetScalar(input_s2w);
    double s2b = mxGetScalar(input_s2b);
    double step_eta = mxGetScalar(input_step_eta);
    double step_kappa = mxGetScalar(input_step_kappa);
    double step_T = mxGetScalar(input_step_T);
    double step_it = mxGetScalar(input_step_it);
    if(step_it==1) {
        step_kappa = 1.0;
    }

    if(Xir==nullptr || Xjc==nullptr) {
        mexErrMsgTxt("The input matrix X must be sparse");
    }
    
    /******************** Allocate memory for output *********************/
    
    /************** Allocate memory for auxiliary variables **************/
    double *grad_w = new double[K];

    /***************************** Main body *****************************/

    // loop over non-zero columns of the observation matrix
    int Yn;
    int Yn_s;
    int Nelem_d;
    int nn;
    double valX;
    double val_dL;
    double new_val;
    double grad;
    double Gt;
    double result;
    int Xjc_dd;
    double eta_pow = step_eta*pow(step_it,-0.5+1.0e-16);

    // Form the set of updated classes k (observed + negsamples)
    std::set<int> classes_updated;
    for(int nn=0; nn<N; nn++){
        Yn = static_cast<int>(Y[nn]);
        classes_updated.insert(Yn);
        for(int ss=0; ss<ns; ss++) {
            Yn_s = static_cast<int>(getdouble_2D(negsamples, nn, ss, N, ns));
            classes_updated.insert(Yn_s);
        }
    }

    // Weight updates
    int kk;
    for(int dd=0; dd<D; dd++) {
        Xjc_dd = Xjc[dd];
        Nelem_d = Xjc[dd+1] - Xjc_dd;
        // initialize gradient to 0
        for(std::set<int>::iterator iter_kk=classes_updated.begin(); iter_kk!=classes_updated.end(); ++iter_kk) {
            kk = *iter_kk;
            grad_w[kk] = 0.0;
        }
        // loop over non-zero rows within that column
        for(int rr=0; rr<Nelem_d; rr++) {
            nn = Xir[Xjc_dd+rr];
            Yn = static_cast<int>(Y[nn]);
            valX = Xval[Xjc_dd+rr];
            if(dLdpsi_botev != nullptr) {
                grad_w[Yn] += (1.0 - dLdpsi_botev[nn]) * valX;
            }
            // loop over negative samples
            for(int ss=0; ss<ns; ss++) {
                Yn_s = static_cast<int>(getdouble_2D(negsamples, nn, ss, N, ns));
                val_dL = getdouble_2D(dLdpsi, nn, ss, N, ns);
                if(dLdpsi_botev != nullptr) {
                    grad_w[Yn_s] -= val_dL * valX;
                } else {
                    result = valX * val_dL;
                    grad_w[Yn] += result;
                    grad_w[Yn_s] -= result;
                }
            }
        }
        // increase values
        for(std::set<int>::iterator iter_kk=classes_updated.begin(); iter_kk!=classes_updated.end(); ++iter_kk) {
            kk = *iter_kk;
            grad = grad_w[kk];
            if(grad!=0.0) {
                grad *= sviFactor;
                if(!mxIsInf(s2w)) {
                    grad -= getdouble_2D(weights, dd, kk, D, K)/s2w;
                }
                Gt = getdouble_2D(Gt_w, dd, kk, D, K);
                Gt = step_kappa*grad*grad + (1.0-step_kappa)*Gt;
                setdouble_2D(Gt, Gt_w, dd, kk, D, K);
                new_val = eta_pow*grad/(step_T+sqrt(Gt));
                incrdouble_2D(new_val, weights, dd, kk, D, K);
            }
        }
    }
    // Biases updates
    Nelem_d = N;
    // initialize gradient to 0
    for(std::set<int>::iterator iter_kk=classes_updated.begin(); iter_kk!=classes_updated.end(); ++iter_kk) {
        kk = *iter_kk;
        grad_w[kk] = 0.0;
    }
    // loop over non-zero rows with active bias (all of them)
    for(int rr=0; rr<Nelem_d; rr++) {
        nn = rr;
        Yn = static_cast<int>(Y[nn]);
        if(dLdpsi_botev != nullptr) {
            grad_w[Yn] += (1.0 - dLdpsi_botev[nn]);
        }
        // loop over negative samples
        for(int ss=0; ss<ns; ss++) {
            Yn_s = static_cast<int>(getdouble_2D(negsamples, nn, ss, N, ns));
            if(dLdpsi_botev != nullptr) {
                grad_w[Yn_s] -= getdouble_2D(dLdpsi, nn, ss, N, ns);
            } else {
                result = getdouble_2D(dLdpsi, nn, ss, N, ns);
                grad_w[Yn] += result;
                grad_w[Yn_s] -= result;
            }
        }
    }
    // increase values
    for(std::set<int>::iterator iter_kk=classes_updated.begin(); iter_kk!=classes_updated.end(); ++iter_kk) {
        kk = *iter_kk;
        grad = grad_w[kk];
        if(grad!=0.0) {
            grad *= sviFactor;
            if(!mxIsInf(s2b)) {
                grad -= biases[kk]/s2b;
            }
            Gt = Gt_b[kk];
            Gt = step_kappa*grad*grad + (1.0-step_kappa)*Gt;
            Gt_b[kk] = Gt;
            new_val = eta_pow*grad/(step_T+sqrt(Gt));
            biases[kk] += new_val;
        }
    }
    
    /**************************** Free memory ****************************/
    delete [] grad_w;
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

inline double my_pow2(double val) {
    return(val*val);
}

/*************************************************************************/

