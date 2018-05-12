#include <math.h>
#include <stdio.h>
#include <iostream>
//#include <time.h>
#include <stdlib.h>
//#include "gsl_headers.h"
#include <matrix.h>
#include <set>


//#include <gsl/gsl_sf_gamma.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_linalg.h>
//#include <gsl/gsl_blas.h>
//#include <gsl/gsl_sf_exp.h>
//#include <gsl/gsl_sf_log.h>
//#include <gsl/gsl_math.h>
//#include "gsl/gsl_cdf.h"
#include "gsl/gsl_randist.h"

#include "mex.h"


//#define myINFINITY 1.0e100

//Methods included in this header file
inline double getdouble_2D(double *x, int n1, int n2, int N1, int N2);
inline void setdouble_2D(double val, double *x, int n1, int n2, int N1, int N2);

// Misc functions
//inline double my_exp(double val);
//inline double my_log(double val);
