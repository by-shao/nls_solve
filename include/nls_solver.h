#ifndef NLS_SOLVER_H
#define NLS_SOLVER_H

#include <stddef.h>

typedef int (*eval_fvec)(
    const void *data,
    size_t m,
    size_t n,
    const double *x,
    double *fvec);

typedef int (*eval_fjac)(
    const void *data,
    size_t m,
    size_t n,
    const double *x,
    double *jac);

typedef enum {
    NLS_ALGO_LM = 0,
    NLS_ALGO_GN,
    NLS_ALGO_MAX
} NlsAlgorithm;

typedef enum {
    LLS_ALGO_CHOLESKY = 0,
    LLS_ALGO_QR,
    LLS_ALGO_MAX
} LlsAlgorithm;

typedef struct {
    eval_fvec f;
    eval_fjac df;

    size_t m;
    size_t n;

    double ftol;
    double xtol;
    double gtol;

    double factor;
    double epsfcn;

    double *diag;

    size_t maxiter;
} nls_problem;

typedef struct nls_solver_s nls_solver;

#define NLS_MAX_ITER           0

#define NLS_SUCCESS_XTOL       1
#define NLS_SUCCESS_FTOL       2
#define NLS_SUCCESS_GTOL       3

#define NLS_ERR_INVALID       -1
#define NLS_ERR_ALLOC         -2
#define NLS_ERR_CALLBACK      -3
#define NLS_ERR_NUMERIC       -4
#define NLS_ERR_LINEAR        -5

int nls_problem_init(
    nls_problem *prob,
    eval_fvec f,
    eval_fjac df,
    size_t m,
    size_t n);

nls_solver *nls_solver_alloc(
    nls_problem *prob,
    NlsAlgorithm nls,
    LlsAlgorithm lls);

int nls_solve(
    nls_solver *solver,
    const void *data,
    double *params);

void nls_solver_free(
    nls_solver *solver);

#endif
