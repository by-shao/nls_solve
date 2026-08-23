#ifndef NLS_INTERNAL_H
#define NLS_INTERNAL_H

#include "nls_solver.h"

struct nls_solver_s {
    nls_problem problem;
    NlsAlgorithm nls_algorithm;
    LlsAlgorithm lls_algorithm;

    double *fvec;
    double *fjac;
    double *jac_row;
    double *diag;
    double *diag_config;
    double *qtf;
    double *wa1;
    double *wa2;
    double *wa3;
    double *wa4;
    int *ipvt;

    double *jtj;
    double *jtf;
    double *step;
};

int nls_gn_solve(nls_solver *solver, const void *data, double *params);

#endif
