#include "auglag_constrain.h"
#include "nls_internal.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int size_add_ok(size_t a, size_t b, size_t *sum)
{
    if (a > SIZE_MAX - b) {
        return 0;
    }
    *sum = a + b;
    return 1;
}

static void *checked_calloc(size_t count, size_t size)
{
    if (count != 0 && size > SIZE_MAX / count) {
        return NULL;
    }
    return calloc(count, size);
}

static int finite_array(const double *values, size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static int nonnegative_finite_array(const double *values, size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i]) || values[i] < 0.0) {
            return 0;
        }
    }
    return 1;
}

static void saturating_size_add(size_t *total, size_t increment)
{
    if (increment > SIZE_MAX - *total) {
        *total = SIZE_MAX;
    } else {
        *total += increment;
    }
}

void auglag_options_init(AugLagOptions *options)
{
    if (options == NULL) {
        return;
    }
    options->rho_init = 1.0;
    options->rho_factor = 10.0;
    options->rho_update_tau = 0.5;
    options->rho_max = 1.0e12;
    options->constraint_tol = 1.0e-8;
    options->max_outer_iter = 50;
}

static int options_valid(const AugLagOptions *options)
{
    return options != NULL &&
        isfinite(options->rho_init) && options->rho_init > 0.0 &&
        isfinite(options->rho_factor) && options->rho_factor > 1.0 &&
        isfinite(options->rho_update_tau) &&
        options->rho_update_tau > 0.0 && options->rho_update_tau < 1.0 &&
        isfinite(options->rho_max) &&
        options->rho_max >= options->rho_init &&
        isfinite(options->constraint_tol) &&
        options->constraint_tol > 0.0 &&
        options->max_outer_iter > 0;
}

static int problem_valid(const AugLagProblem *problem)
{
    return problem != NULL && problem->residual != NULL &&
        problem->jacobian != NULL && problem->m > 0 && problem->n > 0;
}

static int constraint_set_valid(const AugLagConstraintSet *constraint)
{
    size_t i;

    if (constraint == NULL) {
        return 1;
    }
    if ((constraint->neq != 0 && constraint->eq == NULL) ||
        (constraint->nineq != 0 && constraint->ineq == NULL)) {
        return 0;
    }

    for (i = 0; i < constraint->neq; ++i) {
        if (constraint->eq[i].eval == NULL ||
            constraint->eq[i].jac == NULL ||
            !isfinite(constraint->eq[i].tol) ||
            constraint->eq[i].tol < 0.0) {
            return 0;
        }
    }
    for (i = 0; i < constraint->nineq; ++i) {
        if (constraint->ineq[i].eval == NULL ||
            constraint->ineq[i].jac == NULL ||
            !isfinite(constraint->ineq[i].tol) ||
            constraint->ineq[i].tol < 0.0) {
            return 0;
        }
    }
    return 1;
}

static int multiplier_init_valid(const AugLagMultiplierInit *multiplier_init, size_t neq, size_t nineq)
{
    if (multiplier_init == NULL) {
        return 1;
    }
    if (multiplier_init->lambda != NULL &&
        (multiplier_init->lambda_count != neq ||
         !finite_array(multiplier_init->lambda, neq))) {
        return 0;
    }
    if (multiplier_init->mu != NULL &&
        (multiplier_init->mu_count != nineq ||
         !nonnegative_finite_array(multiplier_init->mu, nineq))) {
        return 0;
    }
    return 1;
}

static int copy_constraint_descriptors(AugLagContext *ctx, const AugLagConstraintSet *constraint)
{
    if (constraint->neq != 0) {
        ctx->owned_eq = (AugLagConstraint *)checked_calloc(
            constraint->neq, sizeof(*ctx->owned_eq));
        if (ctx->owned_eq == NULL) {
            return NLS_ERR_ALLOC;
        }
        memcpy(
            ctx->owned_eq,
            constraint->eq,
            constraint->neq * sizeof(*ctx->owned_eq));
    }
    if (constraint->nineq != 0) {
        ctx->owned_ineq = (AugLagConstraint *)checked_calloc(
            constraint->nineq, sizeof(*ctx->owned_ineq));
        if (ctx->owned_ineq == NULL) {
            return NLS_ERR_ALLOC;
        }
        memcpy(
            ctx->owned_ineq,
            constraint->ineq,
            constraint->nineq * sizeof(*ctx->owned_ineq));
    }
    ctx->constraint.eq = ctx->owned_eq;
    ctx->constraint.neq = constraint->neq;
    ctx->constraint.ineq = ctx->owned_ineq;
    ctx->constraint.nineq = constraint->nineq;
    return 0;
}

static int auglag_eval_fvec(const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    const AugLagContext *ctx = (const AugLagContext *)data;
    double sqrt_rho;
    size_t i;

    if (ctx == NULL || x == NULL || fvec == NULL ||
        m != ctx->aug_m || n != ctx->problem.n ||
        ctx->problem.residual == NULL ||
        !isfinite(ctx->rho) || ctx->rho <= 0.0 ||
        (ctx->constraint.neq != 0 && ctx->lambda == NULL) ||
        (ctx->constraint.nineq != 0 && ctx->mu == NULL)) {
        return 1;
    }
    sqrt_rho = sqrt(ctx->rho);
    if (!isfinite(sqrt_rho)) {
        return 1;
    }

    if (ctx->problem.residual(
            ctx->problem.user_data,
            ctx->problem.m,
            ctx->problem.n,
            x,
            fvec) != 0) {
        return 1;
    }
    if (!finite_array(fvec, ctx->problem.m)) {
        return 0;
    }

    for (i = 0; i < ctx->constraint.neq; ++i) {
        double value;

        if (ctx->constraint.eq[i].eval(
                x, &value, ctx->constraint.eq[i].data) != 0) {
            return 1;
        }
        if (!isfinite(value)) {
            fvec[ctx->problem.m + i] = NAN;
            return 0;
        }
        fvec[ctx->problem.m + i] =
            sqrt_rho * (value + ctx->lambda[i] / ctx->rho);
        if (!isfinite(fvec[ctx->problem.m + i])) {
            fvec[ctx->problem.m + i] = NAN;
            return 0;
        }
    }

    for (i = 0; i < ctx->constraint.nineq; ++i) {
        double value;
        double shifted;
        const size_t row = ctx->problem.m + ctx->constraint.neq + i;

        if (ctx->constraint.ineq[i].eval(
                x, &value, ctx->constraint.ineq[i].data) != 0) {
            return 1;
        }
        if (!isfinite(value)) {
            fvec[row] = NAN;
            return 0;
        }
        shifted = value + ctx->mu[i] / ctx->rho;
        if (!isfinite(shifted)) {
            fvec[row] = NAN;
            return 0;
        }
        fvec[row] = sqrt_rho * fmax(0.0, shifted);
        if (!isfinite(fvec[row])) {
            fvec[row] = NAN;
            return 0;
        }
    }

    return 0;
}

static int auglag_eval_fjac(const void *data, size_t m, size_t n, const double *x, double *jac)
{
    const AugLagContext *ctx = (const AugLagContext *)data;
    double sqrt_rho;
    size_t i;
    size_t j;

    if (ctx == NULL || x == NULL || jac == NULL ||
        m != ctx->aug_m || n != ctx->problem.n ||
        ctx->problem.jacobian == NULL ||
        !isfinite(ctx->rho) || ctx->rho <= 0.0 ||
        (ctx->constraint.neq != 0 && ctx->lambda == NULL) ||
        (ctx->constraint.nineq != 0 && ctx->mu == NULL)) {
        return 1;
    }
    sqrt_rho = sqrt(ctx->rho);
    if (!isfinite(sqrt_rho)) {
        return 1;
    }

    if (ctx->problem.jacobian(
            ctx->problem.user_data,
            ctx->problem.m,
            ctx->problem.n,
            x,
            jac) != 0) {
        return 1;
    }
    if (!finite_array(jac, ctx->problem.m * ctx->problem.n)) {
        return 0;
    }

    for (i = 0; i < ctx->constraint.neq; ++i) {
        double *row = jac + (ctx->problem.m + i) * ctx->problem.n;

        if (ctx->constraint.eq[i].jac(
                x, row, ctx->constraint.eq[i].data) != 0) {
            return 1;
        }
        if (!finite_array(row, ctx->problem.n)) {
            return 0;
        }
        for (j = 0; j < ctx->problem.n; ++j) {
            row[j] *= sqrt_rho;
        }
        if (!finite_array(row, ctx->problem.n)) {
            row[0] = NAN;
            return 0;
        }
    }

    for (i = 0; i < ctx->constraint.nineq; ++i) {
        const AugLagConstraint *constraint = &ctx->constraint.ineq[i];
        double *row = jac +
            (ctx->problem.m + ctx->constraint.neq + i) * ctx->problem.n;
        double value;
        double shifted;

        if (constraint->eval(x, &value, constraint->data) != 0) {
            return 1;
        }
        if (!isfinite(value)) {
            row[0] = NAN;
            return 0;
        }
        shifted = value + ctx->mu[i] / ctx->rho;
        if (!isfinite(shifted)) {
            row[0] = NAN;
            return 0;
        }
        if (shifted > 0.0) {
            if (constraint->jac(x, row, constraint->data) != 0) {
                return 1;
            }
            if (!finite_array(row, ctx->problem.n)) {
                return 0;
            }
            for (j = 0; j < ctx->problem.n; ++j) {
                row[j] *= sqrt_rho;
            }
            if (!finite_array(row, ctx->problem.n)) {
                row[0] = NAN;
                return 0;
            }
        } else {
            memset(row, 0, ctx->problem.n * sizeof(*row));
        }
    }

    return 0;
}

static int context_valid(const AugLagContext *ctx, const double *x)
{
    size_t expected_aug_m;

    if (ctx == NULL || x == NULL || !problem_valid(&ctx->problem) ||
        ctx->solver == NULL || !constraint_set_valid(&ctx->constraint) ||
        !options_valid(&ctx->options) ||
        !size_add_ok(ctx->problem.m, ctx->constraint.neq, &expected_aug_m) ||
        !size_add_ok(expected_aug_m, ctx->constraint.nineq, &expected_aug_m) ||
        ctx->aug_m != expected_aug_m ||
        (ctx->constraint.neq != 0 && ctx->lambda == NULL) ||
        (ctx->constraint.nineq != 0 && ctx->mu == NULL) ||
        !isfinite(ctx->rho) || ctx->rho <= 0.0 ||
        ctx->rho > ctx->options.rho_max ||
        isnan(ctx->previous_violation) || ctx->previous_violation < 0.0 ||
        !finite_array(x, ctx->problem.n) ||
        !finite_array(ctx->lambda, ctx->constraint.neq) ||
        !nonnegative_finite_array(ctx->mu, ctx->constraint.nineq)) {
        return 0;
    }
    return 1;
}

static int constraint_violation(const AugLagContext *ctx, const double *x, double *violation)
{
    double maximum = 0.0;
    size_t i;

    for (i = 0; i < ctx->constraint.neq; ++i) {
        double value;

        if (ctx->constraint.eq[i].eval(
                x, &value, ctx->constraint.eq[i].data) != 0) {
            return NLS_ERR_CALLBACK;
        }
        if (!isfinite(value)) {
            return NLS_ERR_NUMERIC;
        }
        maximum = fmax(maximum, fabs(value));
    }

    for (i = 0; i < ctx->constraint.nineq; ++i) {
        double value;

        if (ctx->constraint.ineq[i].eval(
                x, &value, ctx->constraint.ineq[i].data) != 0) {
            return NLS_ERR_CALLBACK;
        }
        if (!isfinite(value)) {
            return NLS_ERR_NUMERIC;
        }
        maximum = fmax(maximum, fmax(0.0, value));
    }

    *violation = maximum;
    return 0;
}

static int update_multipliers(AugLagContext *ctx, const double *x)
{
    size_t i;

    for (i = 0; i < ctx->constraint.neq; ++i) {
        double value;
        double updated;

        if (ctx->constraint.eq[i].eval(
                x, &value, ctx->constraint.eq[i].data) != 0) {
            return NLS_ERR_CALLBACK;
        }
        updated = ctx->lambda[i] + ctx->rho * value;
        if (!isfinite(value) || !isfinite(updated)) {
            return NLS_ERR_NUMERIC;
        }
        ctx->lambda[i] = updated;
    }

    for (i = 0; i < ctx->constraint.nineq; ++i) {
        double value;
        double updated;

        if (ctx->constraint.ineq[i].eval(
                x, &value, ctx->constraint.ineq[i].data) != 0) {
            return NLS_ERR_CALLBACK;
        }
        updated = ctx->mu[i] + ctx->rho * value;
        if (!isfinite(value) || !isfinite(updated)) {
            return NLS_ERR_NUMERIC;
        }
        ctx->mu[i] = fmax(0.0, updated);
    }

    return 0;
}

int auglag_init(AugLagContext *ctx, const AugLagProblem *problem, const AugLagConstraintSet *constraint, const AugLagOptions *options, const AugLagMultiplierInit *multiplier_init, NlsAlgorithm nls, LlsAlgorithm lls)
{
    AugLagConstraintSet empty_constraint = {0};
    AugLagOptions default_options;
    nls_problem inner_problem;
    size_t aug_m;
    int status;

    auglag_options_init(&default_options);
    if (constraint == NULL) {
        constraint = &empty_constraint;
    }
    if (options == NULL) {
        options = &default_options;
    }
    if (ctx == NULL || !problem_valid(problem) ||
        !constraint_set_valid(constraint) || !options_valid(options) ||
        !multiplier_init_valid(
            multiplier_init, constraint->neq, constraint->nineq) ||
        nls < 0 || nls >= NLS_ALGO_MAX ||
        lls < 0 || lls >= LLS_ALGO_MAX ||
        (nls == NLS_ALGO_GN && lls != LLS_ALGO_CHOLESKY)) {
        return NLS_ERR_INVALID;
    }
    if (!size_add_ok(problem->m, constraint->neq, &aug_m) ||
        !size_add_ok(aug_m, constraint->nineq, &aug_m) ||
        aug_m > (size_t)INT_MAX || problem->n > (size_t)INT_MAX ||
        (nls == NLS_ALGO_LM && aug_m < problem->n)) {
        return NLS_ERR_INVALID;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->problem = *problem;
    ctx->options = *options;
    ctx->rho = ctx->options.rho_init;
    ctx->previous_violation = HUGE_VAL;
    ctx->final_constraint_violation = HUGE_VAL;
    ctx->aug_m = aug_m;
    ctx->nls_algorithm = nls;
    ctx->lls_algorithm = lls;

    status = copy_constraint_descriptors(ctx, constraint);
    if (status != 0) {
        auglag_destroy(ctx);
        return status;
    }

    if (constraint->neq != 0) {
        ctx->lambda = (double *)checked_calloc(
            constraint->neq, sizeof(*ctx->lambda));
        if (ctx->lambda == NULL) {
            auglag_destroy(ctx);
            return NLS_ERR_ALLOC;
        }
        if (multiplier_init != NULL && multiplier_init->lambda != NULL) {
            memcpy(
                ctx->lambda,
                multiplier_init->lambda,
                constraint->neq * sizeof(*ctx->lambda));
        }
    }
    if (constraint->nineq != 0) {
        ctx->mu = (double *)checked_calloc(
            constraint->nineq, sizeof(*ctx->mu));
        if (ctx->mu == NULL) {
            auglag_destroy(ctx);
            return NLS_ERR_ALLOC;
        }
        if (multiplier_init != NULL && multiplier_init->mu != NULL) {
            memcpy(
                ctx->mu,
                multiplier_init->mu,
                constraint->nineq * sizeof(*ctx->mu));
        }
    }

    status = nls_problem_init(&inner_problem, auglag_eval_fvec, auglag_eval_fjac, aug_m, problem->n);
    if (status != 0) {
        auglag_destroy(ctx);
        return status;
    }
    ctx->solver = nls_solver_alloc(&inner_problem, nls, lls);
    if (ctx->solver == NULL) {
        auglag_destroy(ctx);
        return NLS_ERR_ALLOC;
    }

    return 0;
}

int auglag_solve(AugLagContext *ctx, double *x)
{
    size_t outer_iteration;

    if (!context_valid(ctx, x)) {
        return NLS_ERR_INVALID;
    }
    ctx->outer_iterations = 0;
    ctx->inner_iterations = 0;
    ctx->inner_function_evaluations = 0;
    ctx->inner_jacobian_evaluations = 0;
    ctx->final_constraint_violation = HUGE_VAL;

    for (outer_iteration = 0;
         outer_iteration < ctx->options.max_outer_iter;
         ++outer_iteration) {
        double current_violation;
        const double previous_violation = ctx->previous_violation;
        nls_solver_stats stats;
        const int inner_status = nls_solve(ctx->solver, ctx, x);
        int status;

        saturating_size_add(&ctx->outer_iterations, 1);
        nls_solver_get_stats(ctx->solver, &stats);
        saturating_size_add(&ctx->inner_iterations, stats.iterations);
        saturating_size_add(&ctx->inner_function_evaluations, stats.nfev);
        saturating_size_add(&ctx->inner_jacobian_evaluations, stats.njev);
        if (inner_status < 0) {
            return inner_status;
        }
        status = constraint_violation(ctx, x, &current_violation);
        if (status != 0) {
            return status;
        }
        ctx->final_constraint_violation = current_violation;
        if (current_violation <= ctx->options.constraint_tol &&
            inner_status != NLS_MAX_ITER) {
            ctx->previous_violation = current_violation;
            return AUGLAG_SUCCESS;
        }

        if (current_violation > ctx->options.constraint_tol) {
            status = update_multipliers(ctx, x);
            if (status != 0) {
                return status;
            }
        }

        if (current_violation >
            ctx->options.rho_update_tau * previous_violation) {
            if (ctx->options.rho_factor > ctx->options.rho_max / ctx->rho) {
                ctx->rho = ctx->options.rho_max;
            } else {
                ctx->rho *= ctx->options.rho_factor;
            }
        }
        ctx->previous_violation = current_violation;
    }

    return AUGLAG_MAX_ITER;
}

void auglag_destroy(AugLagContext *ctx)
{
    if (ctx == NULL) {
        return;
    }

    nls_solver_free(ctx->solver);
    free(ctx->lambda);
    free(ctx->mu);
    free(ctx->owned_eq);
    free(ctx->owned_ineq);
    memset(ctx, 0, sizeof(*ctx));
}
