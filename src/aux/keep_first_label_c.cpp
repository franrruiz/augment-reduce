#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <matrix.h>
#include <set>
#include <map>
#include <vector>
#include "mex.h"

#define input_xTr prhs[0]
#define input_xTs prhs[1]
#define input_Ntr prhs[2]
#define input_Nts prhs[3]
#define input_Kini prhs[4]

#define output_yTr plhs[0]
#define output_yTs plhs[1]

/* INPUTS:
 *
 *  0: xTr,          train labels (Ntr x Kini)
 *  1: xTs,          test labels (Nts x Kini)
 *  2: Ntr,          number of train samples
 *  3: Nts,          number of test samples
 *  4: Kini,         initial number of classes
 *
 */

/* OUTPUTS:
 *
 *  0: yTr,          train labels (Ntr x 1)
 *  1: yTs,          test labels (Nts x 1)
 *
 */

inline double getdouble_2D(double *x, int n1, int n2, int N1, int N2);
inline void setdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);
inline void incrdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);
void set_to_zero(double *p, int N);
void set_to_zero(bool *p, int N);
void add_one(double *p, int N);

void mexFunction( int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[] ) {
    
    /**************************** Read inputs ****************************/
    //double *xTr_val = mxGetPr(input_xTr);
    mwIndex *xTr_ir = mxGetIr(input_xTr);
    mwIndex *xTr_jc = mxGetJc(input_xTr);
    //double *xTs_val = mxGetPr(input_xTs);
    mwIndex *xTs_ir = mxGetIr(input_xTs);
    mwIndex *xTs_jc = mxGetJc(input_xTs);
    int Ntr = mxGetScalar(input_Ntr);
    int Nts = mxGetScalar(input_Nts);
    int Kini = mxGetScalar(input_Kini);
    
    if(xTr_ir==nullptr || xTr_jc==nullptr || xTs_ir==nullptr || xTs_jc==nullptr) {
        mexErrMsgTxt("The input matrices must be sparse");
    }
    
    /******************** Allocate memory for output *********************/
    mwSize *dims = (mwSize*)calloc(2,sizeof(mwSize));
    dims[0] = Ntr;
    dims[1] = 1;
    output_yTr = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
    double *yTr = mxGetPr(output_yTr);
    dims[0] = Nts;
    dims[1] = 1;
    output_yTs = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
    double *yTs = mxGetPr(output_yTs);
    free(dims);
    
    /************** Allocate memory for auxiliary variables **************/


    /***************************** Main body *****************************/

	// initialize output to 0
	set_to_zero(yTr, Ntr);
	set_to_zero(yTs, Nts);

	// run over all non-zero entries of input labels
    bool *label_set = new bool[Kini];
    set_to_zero(label_set, Kini);
    int Nelem_k;
    int xjc_kk;
    int nn;
    for(int kk=0; kk<Kini; kk++) {
    	xjc_kk = xTr_jc[kk];
    	Nelem_k = xTr_jc[kk+1] - xjc_kk;
    	// loop over all datapoints that have class kk
        for(int rr=0; rr<Nelem_k; rr++) {
            nn = xTr_ir[xjc_kk+rr];
            // if class has not been assigned yet, set class
            if(yTr[nn]==0) {
            	yTr[nn] = kk+1;
            	label_set[kk] = true;
            }
		}
    }

    bool flag0 = false;
    for(nn=0; nn<Ntr; nn++) {
    	if(yTr[nn]==0) {
    		flag0 = true;
    		break;
    	}
    }
    if(flag0) {
	    add_one(yTr, Ntr);
    }

	// run over all non-zero entries of input labels (test set)
    for(int kk=0; kk<Kini; kk++) {
    	xjc_kk = xTs_jc[kk];
    	Nelem_k = xTs_jc[kk+1] - xjc_kk;
    	// loop over all datapoints that have class kk
        for(int rr=0; rr<Nelem_k; rr++) {
            nn = xTs_ir[xjc_kk+rr];
            // if class has not been assigned yet and it is a valid label, set class
            if(yTs[nn]==0 && label_set[kk]) {
        		yTs[nn] = kk+1;
            }
		}
    }
    if(flag0) {
	    add_one(yTs, Nts);
	}

    // Create map from labels
    std::map<int,int> labelmap;
    int count = 1;
    for(nn=0; nn<Ntr; nn++) {
        std::map<int,int>::iterator iter_m = labelmap.find(static_cast<int>(yTr[nn]));
        if(iter_m==labelmap.end()) {
            labelmap.insert(std::pair<int,int>(static_cast<int>(yTr[nn]), count));
            yTr[nn] = count;
            count++;
        } else {
            yTr[nn] = iter_m->second;
        }
    }
    for(nn=0; nn<Nts; nn++) {
        std::map<int,int>::iterator iter_m = labelmap.find(static_cast<int>(yTs[nn]));
        if(iter_m==labelmap.end()) {
            yTs[nn] = -1;
        } else {
            yTs[nn] = iter_m->second;
        }
    }

    /***************************** Set output *****************************/

    /**************************** Free memory ****************************/
    delete [] label_set;
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

void add_one(double *p, int N) {
	for(int nn=0; nn<N; nn++) {
		p[nn] += 1;
	}
}

void set_to_zero(double *p, int N) {
	for(int nn=0; nn<N; nn++) {
		p[nn] = 0;
	}
}

void set_to_zero(bool *p, int N) {
	for(int nn=0; nn<N; nn++) {
		p[nn] = false;
	}
}

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

