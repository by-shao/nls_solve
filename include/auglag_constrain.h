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
    /* Descriptor arrays are copied by auglag_init; each data pointer is borrowed. */
    const AugLagConstraint *eq;
    size_t neq;

    const AugLagConstraint *ineq;
    size_t nineq;
} AugLagConstraintSet;

typedef struct {
    eval_fvec residual;
    eval_fjac jacobian;
    void *user_data;
    size_t m;
    size_t n;
} AugLagProblem;

typedef struct {
    double rho_init;
    double rho_factor;
    double rho_update_tau;
    double rho_max;
    double constraint_tol;
    size_t max_outer_iter;
} AugLagOptions;

typedef struct {
    const double *lambda;
    size_t lambda_count;
    const double *mu;
    size_t mu_count;
} AugLagMultiplierInit;

typedef struct {
    /* Problem descriptors are copied. Callback data pointers remain borrowed. */
    AugLagProblem problem;
    AugLagConstraintSet constraint;
    AugLagOptions options;

    double *lambda;
    double *mu;

    /* Runtime state. options.rho_init is never modified by the solver. */
    double rho;
    double previous_violation;
    double final_constraint_violation;

    size_t aug_m;
    nls_solver *solver;
    NlsAlgorithm nls_algorithm;
    LlsAlgorithm lls_algorithm;

    /* Statistics from the most recent auglag_solve call. */
    size_t outer_iterations;
    /* Exact GN iterations; zero for LM, whose available metric is nfev. */
    size_t inner_iterations;
    size_t inner_function_evaluations;
    size_t inner_jacobian_evaluations;

    /* Owned copies of the constraint descriptors. */
    AugLagConstraint *owned_eq;
    AugLagConstraint *owned_ineq;
} AugLagContext;

#define AUGLAG_MAX_ITER 0
#define AUGLAG_SUCCESS  1

void auglag_options_init(AugLagOptions *options);

int auglag_init(AugLagContext *ctx, const AugLagProblem *problem, const AugLagConstraintSet *constraint, const AugLagOptions *options, const AugLagMultiplierInit *multiplier_init, NlsAlgorithm nls, LlsAlgorithm lls);

int auglag_solve(AugLagContext *ctx, double *x);

void auglag_destroy(AugLagContext *ctx);

#endif
