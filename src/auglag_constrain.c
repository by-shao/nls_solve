#include "auglag_constrain.h"

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

static int auglag_eval_fvec(
    const void *data,
    size_t m,
    size_t n,
    const double *x,
    double *fvec)
{
    const AugLagContext *ctx = (const AugLagContext *)data;
    const double sqrt_rho =
        ctx != NULL && ctx->rho > 0.0 ? sqrt(ctx->rho) : 0.0;
    size_t i;

    if (ctx == NULL || x == NULL || fvec == NULL ||
        m != ctx->aug_m || n != ctx->n || ctx->residual == NULL ||
        !isfinite(ctx->rho) || ctx->rho <= 0.0 ||
        !isfinite(sqrt_rho) ||
        (ctx->constraint.neq != 0 && ctx->lambda == NULL) ||
        (ctx->constraint.nineq != 0 && ctx->mu == NULL)) {
        return 1;
    }

    if (ctx->residual(
            ctx->user_data, ctx->m, ctx->n, x, fvec) != 0) {
        return 1;
    }

    for (i = 0; i < ctx->constraint.neq; ++i) {
        double value;

        if (ctx->constraint.eq[i].eval(
                x, &value, ctx->constraint.eq[i].data) != 0) {
            return 1;
        }
        fvec[ctx->m + i] =
            sqrt_rho * (value + ctx->lambda[i] / ctx->rho);
    }

    for (i = 0; i < ctx->constraint.nineq; ++i) {
        double value;
        double shifted;

        if (ctx->constraint.ineq[i].eval(
                x, &value, ctx->constraint.ineq[i].data) != 0) {
            return 1;
        }
        shifted = value + ctx->mu[i] / ctx->rho;
        fvec[ctx->m + ctx->constraint.neq + i] =
            sqrt_rho * fmax(0.0, shifted);
    }

    return 0;
}

static int auglag_eval_fjac(
    const void *data,
    size_t m,
    size_t n,
    const double *x,
    double *jac)
{
    const AugLagContext *ctx = (const AugLagContext *)data;
    const double sqrt_rho =
        ctx != NULL && ctx->rho > 0.0 ? sqrt(ctx->rho) : 0.0;
    size_t i;
    size_t j;

    if (ctx == NULL || x == NULL || jac == NULL ||
        m != ctx->aug_m || n != ctx->n || ctx->jacobian == NULL ||
        !isfinite(ctx->rho) || ctx->rho <= 0.0 ||
        !isfinite(sqrt_rho) ||
        (ctx->constraint.neq != 0 && ctx->lambda == NULL) ||
        (ctx->constraint.nineq != 0 && ctx->mu == NULL)) {
        return 1;
    }

    if (ctx->jacobian(
            ctx->user_data, ctx->m, ctx->n, x, jac) != 0) {
        return 1;
    }

    for (i = 0; i < ctx->constraint.neq; ++i) {
        double *row = jac + (ctx->m + i) * ctx->n;

        if (ctx->constraint.eq[i].jac(
                x, row, ctx->constraint.eq[i].data) != 0) {
            return 1;
        }
        for (j = 0; j < ctx->n; ++j) {
            row[j] *= sqrt_rho;
        }
    }

    for (i = 0; i < ctx->constraint.nineq; ++i) {
        const AugLagConstraint *constraint = &ctx->constraint.ineq[i];
        double *row =
            jac + (ctx->m + ctx->constraint.neq + i) * ctx->n;
        double value;
        double shifted;

        if (constraint->eval(x, &value, constraint->data) != 0) {
            return 1;
        }
        shifted = value + ctx->mu[i] / ctx->rho;
        if (!isfinite(shifted)) {
            for (j = 0; j < ctx->n; ++j) {
                row[j] = NAN;
            }
        } else if (shifted > 0.0) {
            if (constraint->jac(x, row, constraint->data) != 0) {
                return 1;
            }
            for (j = 0; j < ctx->n; ++j) {
                row[j] *= sqrt_rho;
            }
        } else {
            memset(row, 0, ctx->n * sizeof(*row));
        }
    }

    return 0;
}

static int context_valid(const AugLagContext *ctx, const double *x)
{
    size_t expected_aug_m;

    if (ctx == NULL || x == NULL || ctx->residual == NULL ||
        ctx->jacobian == NULL || ctx->m == 0 || ctx->n == 0 ||
        ctx->solver == NULL || !constraint_set_valid(&ctx->constraint) ||
        !size_add_ok(ctx->m, ctx->constraint.neq, &expected_aug_m) ||
        !size_add_ok(expected_aug_m, ctx->constraint.nineq, &expected_aug_m) ||
        ctx->aug_m != expected_aug_m ||
        (ctx->constraint.neq != 0 && ctx->lambda == NULL) ||
        (ctx->constraint.nineq != 0 && ctx->mu == NULL) ||
        !isfinite(ctx->rho) || ctx->rho <= 0.0 ||
        !isfinite(ctx->rho_factor) || ctx->rho_factor < 1.0 ||
        !isfinite(ctx->rho_update_tau) || ctx->rho_update_tau < 0.0 ||
        ctx->rho_update_tau > 1.0 ||
        !isfinite(ctx->rho_max) || ctx->rho_max < ctx->rho ||
        !isfinite(ctx->constraint_tol) || ctx->constraint_tol < 0.0 ||
        ctx->max_outer_iter == 0 || isnan(ctx->previous_violation) ||
        ctx->previous_violation < 0.0 || !finite_array(x, ctx->n) ||
        !finite_array(ctx->lambda, ctx->constraint.neq) ||
        !finite_array(ctx->mu, ctx->constraint.nineq)) {
        return 0;
    }
    return 1;
}

static int constraint_violation(
    const AugLagContext *ctx,
    const double *x,
    double *violation)
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

int auglag_init(
    AugLagContext *ctx,
    eval_fvec residual,
    eval_fjac jacobian,
    void *user_data,
    size_t m,
    size_t n,
    const AugLagConstraintSet *constraint,
    NlsAlgorithm nls,
    LlsAlgorithm lls)
{
    AugLagConstraintSet empty_constraint = {0};
    nls_problem problem;
    size_t aug_m;
    int status;

    if (ctx == NULL || residual == NULL || jacobian == NULL ||
        m == 0 || n == 0 || !constraint_set_valid(constraint) ||
        nls < 0 || nls >= NLS_ALGO_MAX ||
        lls < 0 || lls >= LLS_ALGO_MAX ||
        (nls == NLS_ALGO_GN && lls != LLS_ALGO_CHOLESKY)) {
        return NLS_ERR_INVALID;
    }
    if (constraint == NULL) {
        constraint = &empty_constraint;
    }
    if (!size_add_ok(m, constraint->neq, &aug_m) ||
        !size_add_ok(aug_m, constraint->nineq, &aug_m) ||
        aug_m > (size_t)INT_MAX || n > (size_t)INT_MAX ||
        (nls == NLS_ALGO_LM && aug_m < n)) {
        return NLS_ERR_INVALID;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->residual = residual;
    ctx->jacobian = jacobian;
    ctx->user_data = user_data;
    ctx->m = m;
    ctx->n = n;
    ctx->constraint = *constraint;
    ctx->rho = 1.0;
    ctx->rho_factor = 10.0;
    ctx->rho_update_tau = 0.5;
    ctx->rho_max = 1.0e12;
    ctx->constraint_tol = 1.0e-8;
    ctx->max_outer_iter = 50;
    ctx->previous_violation = HUGE_VAL;
    ctx->aug_m = aug_m;

    if (constraint->neq != 0) {
        ctx->lambda = (double *)checked_calloc(
            constraint->neq, sizeof(*ctx->lambda));
        if (ctx->lambda == NULL) {
            auglag_destroy(ctx);
            return NLS_ERR_ALLOC;
        }
    }
    if (constraint->nineq != 0) {
        ctx->mu = (double *)checked_calloc(
            constraint->nineq, sizeof(*ctx->mu));
        if (ctx->mu == NULL) {
            auglag_destroy(ctx);
            return NLS_ERR_ALLOC;
        }
    }

    status = nls_problem_init(
        &problem, auglag_eval_fvec, auglag_eval_fjac, aug_m, n);
    if (status != 0) {
        auglag_destroy(ctx);
        return status;
    }
    ctx->solver = nls_solver_alloc(&problem, nls, lls);
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

    for (outer_iteration = 0;
         outer_iteration < ctx->max_outer_iter;
         ++outer_iteration) {
        double current_violation;
        const double previous_violation = ctx->previous_violation;
        int status = nls_solve(ctx->solver, ctx, x);

        if (status < 0) {
            return status;
        }
        status = constraint_violation(ctx, x, &current_violation);
        if (status != 0) {
            return status;
        }
        if (current_violation <= ctx->constraint_tol) {
            ctx->previous_violation = current_violation;
            return AUGLAG_SUCCESS;
        }

        status = update_multipliers(ctx, x);
        if (status != 0) {
            return status;
        }

        if (current_violation >
            ctx->rho_update_tau * previous_violation) {
            if (ctx->rho_factor > ctx->rho_max / ctx->rho) {
                ctx->rho = ctx->rho_max;
            } else {
                ctx->rho *= ctx->rho_factor;
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
    memset(ctx, 0, sizeof(*ctx));
}
