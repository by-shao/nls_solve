#ifndef AUGLAG_CONSTRAIN_H
#define AUGLAG_CONSTRAIN_H

#include "nls_solver.h"

typedef int (*auglag_constraint_eval)(
    const double *x,
    double *value,
    void *data);

typedef int (*auglag_constraint_jac)(
    const double *x,
    double *jac,
    void *data);

typedef struct {
    auglag_constraint_eval eval;
    auglag_constraint_jac jac;
    void *data;
    double tol;
} AugLagConstraint;

typedef struct {
    AugLagConstraint *eq;
    size_t neq;

    AugLagConstraint *ineq;
    size_t nineq;
} AugLagConstraintSet;

typedef struct {
    eval_fvec residual;
    eval_fjac jacobian;

    void *user_data;

    size_t m;
    size_t n;

    AugLagConstraintSet constraint;

    double *lambda;
    double *mu;

    double rho;
    double rho_factor;
    double rho_update_tau;
    double rho_max;

    double constraint_tol;

    size_t max_outer_iter;

    double previous_violation;

    size_t aug_m;

    nls_solver *solver;
} AugLagContext;

#define AUGLAG_MAX_ITER 0
#define AUGLAG_SUCCESS  1

int auglag_init(
    AugLagContext *ctx,
    eval_fvec residual,
    eval_fjac jacobian,
    void *user_data,
    size_t m,
    size_t n,
    const AugLagConstraintSet *constraint,
    NlsAlgorithm nls,
    LlsAlgorithm lls);

int auglag_solve(AugLagContext *ctx, double *x);

void auglag_destroy(AugLagContext *ctx);

#endif
