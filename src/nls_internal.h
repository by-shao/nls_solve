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

    size_t last_iterations;
    size_t last_nfev;
    size_t last_njev;
};

typedef struct {
    size_t iterations;
    size_t nfev;
    size_t njev;
} nls_solver_stats;

int nls_gn_solve(nls_solver *solver, const void *data, double *params);

void nls_solver_get_stats(const nls_solver *solver, nls_solver_stats *stats);

#endif
