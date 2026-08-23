#include "auglag_constrain.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    GAUSSIAN_NX = 31,
    GAUSSIAN_NY = 31,
    GAUSSIAN_COUNT = GAUSSIAN_NX * GAUSSIAN_NY,
    GAUSSIAN_PARAMETER_COUNT = 7,
    GAUSSIAN_POSITIVE_BOUND_COUNT = 4,
    GAUSSIAN_MAX_BOUND_COUNT = 5,
    GAUSSIAN_MAX_ACTIVE_BOUND_COUNT = 2
};

typedef struct {
    size_t count;
    const double *x;
    const double *y;
    const double *z;
    const double *outer_params;
    size_t *outer_iterations;
} GaussianFitData;

typedef enum {
    PARAMETER_LOWER_BOUND,
    PARAMETER_UPPER_BOUND
} ParameterBoundKind;

typedef struct {
    size_t index;
    double target;
    ParameterBoundKind kind;
} ParameterBoundData;

typedef struct {
    const char *name;
    const char *special_condition;
    const char *reference_expectation;
    const double *truth;
    const double *initial;
    const ParameterBoundData *bounds;
    size_t bound_count;
    const char *bound_target_text;
    size_t nx;
    size_t ny;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double noise_scale;
    double constraint_tol;
    size_t max_outer_iter;
    double max_rmse;
    double max_prediction_error;
    double max_relative_prediction_rmse;
    double max_jac_error;
    const size_t *expected_active_bounds;
    size_t expected_active_bound_count;
    int require_initial_infeasible;
    int require_box_interior;
    int require_truth_recovery;
    int require_equal_widths;
    int expected_failure;
    int narrow_fd_steps;
    int print_human;
} GaussianCase;

typedef struct {
    double params[GAUSSIAN_PARAMETER_COUNT];
    size_t m;
    size_t m_aug;
    size_t outer_callback_count;
    int status;
    int failed;
    int finite_check;
    double jac_error;
    double initial_violation;
    double rmse;
    double prediction_rmse;
    double max_prediction_error;
    double relative_prediction_rmse;
    double violation;
    double rho;
    double rho_max;
    double parameter_scale_ratio;
    double theta_jacobian_norm;
    size_t underflow_zero_count;
    double active_bound_values[GAUSSIAN_MAX_ACTIVE_BOUND_COUNT];
    double active_bound_targets[GAUSSIAN_MAX_ACTIVE_BOUND_COUNT];
    double active_bound_slacks[GAUSSIAN_MAX_ACTIVE_BOUND_COUNT];
    size_t active_bound_parameters[GAUSSIAN_MAX_ACTIVE_BOUND_COUNT];
    ParameterBoundKind active_bound_kinds[GAUSSIAN_MAX_ACTIVE_BOUND_COUNT];
} GaussianResult;

static double gaussian_model(const double *c, double x, double y)
{
    double dx = x - c[5];
    double dy = y - c[6];
    double ct = cos(c[4]);
    double st = sin(c[4]);
    double u = ct * dx + st * dy;
    double v = -st * dx + ct * dy;
    double q = u * u / (c[2] * c[2]) + v * v / (c[3] * c[3]);

    return c[0] + c[1] * exp(-0.5 * q);
}

static int gaussian_residual(
    const void *data, size_t m, size_t n, const double *params, double *fvec)
{
    const GaussianFitData *fit = (const GaussianFitData *)data;
    size_t i;

    if (fit == NULL || params == NULL || fvec == NULL ||
        fit->x == NULL || fit->y == NULL || fit->z == NULL ||
        m != fit->count || n != GAUSSIAN_PARAMETER_COUNT ||
        params[2] == 0.0 || params[3] == 0.0) {
        return 1;
    }
    for (i = 0; i < n; ++i) {
        if (!isfinite(params[i])) {
            return 1;
        }
    }
    if (fit->outer_params == params && fit->outer_iterations != NULL) {
        ++(*fit->outer_iterations);
    }

    for (i = 0; i < m; ++i) {
        fvec[i] = gaussian_model(params, fit->x[i], fit->y[i]) - fit->z[i];
    }
    return 0;
}

static int gaussian_jacobian(
    const void *data, size_t m, size_t n, const double *params, double *jac)
{
    const GaussianFitData *fit = (const GaussianFitData *)data;
    double amplitude;
    double sx;
    double sy;
    double ct;
    double st;
    double inv_sx2;
    double inv_sy2;
    size_t i;

    if (fit == NULL || params == NULL || jac == NULL ||
        fit->x == NULL || fit->y == NULL || fit->z == NULL ||
        m != fit->count || n != GAUSSIAN_PARAMETER_COUNT ||
        params[2] == 0.0 || params[3] == 0.0) {
        return 1;
    }
    for (i = 0; i < n; ++i) {
        if (!isfinite(params[i])) {
            return 1;
        }
    }

    amplitude = params[1];
    sx = params[2];
    sy = params[3];
    ct = cos(params[4]);
    st = sin(params[4]);
    inv_sx2 = 1.0 / (sx * sx);
    inv_sy2 = 1.0 / (sy * sy);

    for (i = 0; i < m; ++i) {
        double dx = fit->x[i] - params[5];
        double dy = fit->y[i] - params[6];
        double u = ct * dx + st * dy;
        double v = -st * dx + ct * dy;
        double q = u * u * inv_sx2 + v * v * inv_sy2;
        double e = exp(-0.5 * q);
        double *row = jac + i * n;

        row[0] = 1.0;
        row[1] = e;
        row[2] = amplitude * e * u * u / (sx * sx * sx);
        row[3] = amplitude * e * v * v / (sy * sy * sy);
        row[4] = amplitude * e * u * v * (inv_sy2 - inv_sx2);
        row[5] = amplitude * e *
                 (u * ct * inv_sx2 - v * st * inv_sy2);
        row[6] = amplitude * e *
                 (u * st * inv_sx2 + v * ct * inv_sy2);
    }
    return 0;
}

static int parameter_bound_eval(const double *x, double *value, void *data)
{
    const ParameterBoundData *bound = (const ParameterBoundData *)data;

    if (x == NULL || value == NULL || bound == NULL ||
        bound->index >= GAUSSIAN_PARAMETER_COUNT) {
        return 1;
    }
    if (bound->kind == PARAMETER_LOWER_BOUND) {
        *value = bound->target - x[bound->index];
    } else if (bound->kind == PARAMETER_UPPER_BOUND) {
        *value = x[bound->index] - bound->target;
    } else {
        return 1;
    }
    return 0;
}

static int parameter_bound_jac(const double *x, double *jac, void *data)
{
    const ParameterBoundData *bound = (const ParameterBoundData *)data;
    size_t j;

    if (x == NULL || jac == NULL || bound == NULL ||
        bound->index >= GAUSSIAN_PARAMETER_COUNT) {
        return 1;
    }
    for (j = 0; j < GAUSSIAN_PARAMETER_COUNT; ++j) {
        jac[j] = 0.0;
    }
    if (bound->kind == PARAMETER_LOWER_BOUND) {
        jac[bound->index] = -1.0;
    } else if (bound->kind == PARAMETER_UPPER_BOUND) {
        jac[bound->index] = 1.0;
    } else {
        return 1;
    }
    return 0;
}

static int gaussian_jacobian_fd_check(
    GaussianFitData *fit,
    const double *params,
    int narrow_steps,
    double *max_error)
{
    double *analytic = NULL;
    double *fplus = NULL;
    double *fminus = NULL;
    double plus[GAUSSIAN_PARAMETER_COUNT];
    double minus[GAUSSIAN_PARAMETER_COUNT];
    size_t i;
    size_t j;
    int status = 1;

    if (fit == NULL || params == NULL || max_error == NULL ||
        fit->count == 0 ||
        fit->count > SIZE_MAX /
            (GAUSSIAN_PARAMETER_COUNT * sizeof(*analytic))) {
        return 1;
    }
    analytic = (double *)malloc(
        fit->count * GAUSSIAN_PARAMETER_COUNT * sizeof(*analytic));
    fplus = (double *)malloc(fit->count * sizeof(*fplus));
    fminus = (double *)malloc(fit->count * sizeof(*fminus));
    if (analytic == NULL || fplus == NULL || fminus == NULL ||
        gaussian_jacobian(fit,
                          fit->count,
                          GAUSSIAN_PARAMETER_COUNT,
                          params,
                          analytic) != 0) {
        goto cleanup;
    }

    *max_error = 0.0;
    for (j = 0; j < GAUSSIAN_PARAMETER_COUNT; ++j) {
        double step = 1e-6 * fmax(1.0, fabs(params[j]));

        if (narrow_steps && (j == 2 || j == 3)) {
            step = 1e-7 * fmax(fabs(params[j]), 1e-3);
        }
        if (!isfinite(step) || step <= 0.0) {
            goto cleanup;
        }
        for (i = 0; i < GAUSSIAN_PARAMETER_COUNT; ++i) {
            plus[i] = params[i];
            minus[i] = params[i];
        }
        plus[j] += step;
        minus[j] -= step;
        if (gaussian_residual(fit,
                              fit->count,
                              GAUSSIAN_PARAMETER_COUNT,
                              plus,
                              fplus) != 0 ||
            gaussian_residual(fit,
                              fit->count,
                              GAUSSIAN_PARAMETER_COUNT,
                              minus,
                              fminus) != 0) {
            goto cleanup;
        }
        for (i = 0; i < fit->count; ++i) {
            double finite_difference =
                (fplus[i] - fminus[i]) / (2.0 * step);
            double analytic_value =
                analytic[i * GAUSSIAN_PARAMETER_COUNT + j];
            double scale = fmax(
                1.0, fmax(fabs(analytic_value), fabs(finite_difference)));
            double error = fabs(analytic_value - finite_difference) / scale;

            if (!isfinite(analytic_value) ||
                !isfinite(finite_difference) || !isfinite(error)) {
                goto cleanup;
            }
            if (error > *max_error) {
                *max_error = error;
            }
        }
    }
    status = 0;

cleanup:
    free(analytic);
    free(fplus);
    free(fminus);
    return status;
}

static void print_gaussian_params(const char *label, const double *params)
{
    size_t j;

    printf("%s=[", label);
    for (j = 0; j < GAUSSIAN_PARAMETER_COUNT; ++j) {
        printf("%s%.12g", j == 0 ? "" : ", ", params[j]);
    }
    puts("]");
}

static int quadratic_residual(
    const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    (void)data;
    (void)m;
    (void)n;
    fvec[0] = x[0] - 2.0;
    fvec[1] = x[1] - 3.0;
    return 0;
}

static int quadratic_jacobian(
    const void *data, size_t m, size_t n, const double *x, double *jac)
{
    (void)data;
    (void)m;
    (void)n;
    (void)x;
    jac[0] = 1.0;
    jac[1] = 0.0;
    jac[2] = 0.0;
    jac[3] = 1.0;
    return 0;
}

static int scalar_residual(
    const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    (void)data;
    (void)m;
    (void)n;
    fvec[0] = x[0];
    return 0;
}

static int scalar_jacobian(
    const void *data, size_t m, size_t n, const double *x, double *jac)
{
    (void)data;
    (void)m;
    (void)n;
    (void)x;
    jac[0] = 1.0;
    return 0;
}

static int sum_eq_six(const double *x, double *value, void *data)
{
    (void)data;
    *value = x[0] + x[1] - 6.0;
    return 0;
}

static int sum_eq_six_jac(const double *x, double *jac, void *data)
{
    (void)x;
    (void)data;
    jac[0] = 1.0;
    jac[1] = 1.0;
    return 0;
}

static int lower_one(const double *x, double *value, void *data)
{
    (void)data;
    *value = 1.0 - x[0];
    return 0;
}

static int lower_one_jac(const double *x, double *jac, void *data)
{
    (void)x;
    (void)data;
    jac[0] = -1.0;
    return 0;
}

static int lower_three(const double *x, double *value, void *data)
{
    (void)data;
    *value = 3.0 - x[0];
    return 0;
}

static int lower_three_jac(const double *x, double *jac, void *data)
{
    (void)x;
    (void)data;
    jac[0] = -1.0;
    jac[1] = 0.0;
    return 0;
}

static double constraint_violation(
    const AugLagConstraintSet *constraints, const double *x)
{
    double violation = 0.0;
    size_t i;

    for (i = 0; i < constraints->neq; ++i) {
        double value;
        double magnitude;
        if (constraints->eq[i].eval(
                x, &value, constraints->eq[i].data) != 0) {
            return HUGE_VAL;
        }
        magnitude = fabs(value);
        if (magnitude > violation) {
            violation = magnitude;
        }
    }
    for (i = 0; i < constraints->nineq; ++i) {
        double value;
        if (constraints->ineq[i].eval(
                x, &value, constraints->ineq[i].data) != 0) {
            return HUGE_VAL;
        }
        if (value > violation) {
            violation = value;
        }
    }
    return violation;
}

static int report_case(
    const char *name,
    int status,
    const double *x,
    size_t n,
    double violation,
    int failed)
{
    if (n == 1) {
        printf("%s: status=%d x=[%.12g] violation=%.3g %s\n",
               name,
               status,
               x[0],
               violation,
               failed ? "FAIL" : "PASS");
    } else {
        printf("%s: status=%d x=[%.12g, %.12g] violation=%.3g %s\n",
               name,
               status,
               x[0],
               x[1],
               violation,
               failed ? "FAIL" : "PASS");
    }
    return failed;
}

static int test_unconstrained(void)
{
    AugLagContext ctx;
    AugLagConstraintSet constraints = {NULL, 0, NULL, 0};
    double x[2] = {0.0, 0.0};
    double violation;
    int status;
    int failed;

    if (auglag_init(&ctx,
                    quadratic_residual,
                    quadratic_jacobian,
                    NULL,
                    2,
                    2,
                    &constraints,
                    NLS_ALGO_LM,
                    LLS_ALGO_CHOLESKY) != 0) {
        return report_case("CASE 1", NLS_ERR_ALLOC, x, 2, HUGE_VAL, 1);
    }
    ctx.constraint_tol = 1e-8;
    status = auglag_solve(&ctx, x);
    violation = constraint_violation(&constraints, x);
    failed = status != AUGLAG_SUCCESS || fabs(x[0] - 2.0) >= 1e-8 ||
             fabs(x[1] - 3.0) >= 1e-8 || violation > ctx.constraint_tol;
    auglag_destroy(&ctx);
    return report_case("CASE 1", status, x, 2, violation, failed);
}

static int test_equality(void)
{
    AugLagConstraint eq[] = {
        {sum_eq_six, sum_eq_six_jac, NULL, 1e-8},
    };
    AugLagConstraintSet constraints = {eq, 1, NULL, 0};
    AugLagContext ctx;
    double x[2] = {0.0, 0.0};
    double violation;
    int status;
    int failed;

    if (auglag_init(&ctx,
                    quadratic_residual,
                    quadratic_jacobian,
                    NULL,
                    2,
                    2,
                    &constraints,
                    NLS_ALGO_LM,
                    LLS_ALGO_CHOLESKY) != 0) {
        return report_case("CASE 2", NLS_ERR_ALLOC, x, 2, HUGE_VAL, 1);
    }
    ctx.constraint_tol = 1e-8;
    status = auglag_solve(&ctx, x);
    violation = constraint_violation(&constraints, x);
    failed = status != AUGLAG_SUCCESS || fabs(x[0] - 2.5) >= 1e-7 ||
             fabs(x[1] - 3.5) >= 1e-7 || violation > ctx.constraint_tol;
    auglag_destroy(&ctx);
    return report_case("CASE 2", status, x, 2, violation, failed);
}

static int test_inequality(void)
{
    AugLagConstraint ineq[] = {
        {lower_one, lower_one_jac, NULL, 1e-8},
    };
    AugLagConstraintSet constraints = {NULL, 0, ineq, 1};
    AugLagContext ctx;
    double x[1] = {0.0};
    double violation;
    int status;
    int failed;

    if (auglag_init(&ctx,
                    scalar_residual,
                    scalar_jacobian,
                    NULL,
                    1,
                    1,
                    &constraints,
                    NLS_ALGO_LM,
                    LLS_ALGO_CHOLESKY) != 0) {
        return report_case("CASE 3", NLS_ERR_ALLOC, x, 1, HUGE_VAL, 1);
    }
    ctx.constraint_tol = 1e-8;
    status = auglag_solve(&ctx, x);
    violation = constraint_violation(&constraints, x);
    failed = status != AUGLAG_SUCCESS || fabs(x[0] - 1.0) >= 5e-8 ||
             violation > ctx.constraint_tol;
    auglag_destroy(&ctx);
    return report_case("CASE 3", status, x, 1, violation, failed);
}

static int test_mixed_constraints(void)
{
    AugLagConstraint eq[] = {
        {sum_eq_six, sum_eq_six_jac, NULL, 1e-8},
    };
    AugLagConstraint ineq[] = {
        {lower_three, lower_three_jac, NULL, 1e-8},
    };
    AugLagConstraintSet constraints = {eq, 1, ineq, 1};
    AugLagContext ctx;
    double x[2] = {0.0, 0.0};
    double violation;
    int status;
    int failed;

    if (auglag_init(&ctx,
                    quadratic_residual,
                    quadratic_jacobian,
                    NULL,
                    2,
                    2,
                    &constraints,
                    NLS_ALGO_LM,
                    LLS_ALGO_CHOLESKY) != 0) {
        return report_case("CASE 4", NLS_ERR_ALLOC, x, 2, HUGE_VAL, 1);
    }
    ctx.constraint_tol = 1e-8;
    status = auglag_solve(&ctx, x);
    violation = constraint_violation(&constraints, x);
    failed = status != AUGLAG_SUCCESS || fabs(x[0] - 3.0) >= 1e-6 ||
             fabs(x[1] - 3.0) >= 1e-6 || violation > ctx.constraint_tol;
    auglag_destroy(&ctx);
    return report_case("CASE 4", status, x, 2, violation, failed);
}

static double parameter_bound_slack(
    const ParameterBoundData *bound, const double *params)
{
    if (bound->kind == PARAMETER_LOWER_BOUND) {
        return params[bound->index] - bound->target;
    }
    return bound->target - params[bound->index];
}

static int finite_values(const double *values, size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static double gaussian_exponential(const double *params, double x, double y)
{
    double dx = x - params[5];
    double dy = y - params[6];
    double ct = cos(params[4]);
    double st = sin(params[4]);
    double u = ct * dx + st * dy;
    double v = -st * dx + ct * dy;
    double q = u * u / (params[2] * params[2]) +
               v * v / (params[3] * params[3]);

    return exp(-0.5 * q);
}

static double gaussian_parameter_scale_ratio(const double *params)
{
    double maximum = 0.0;
    double minimum = DBL_MAX;
    size_t i;

    for (i = 0; i < GAUSSIAN_PARAMETER_COUNT; ++i) {
        double magnitude = fabs(params[i]);

        if (magnitude > maximum) {
            maximum = magnitude;
        }
        if (magnitude > 0.0 && magnitude < minimum) {
            minimum = magnitude;
        }
    }
    if (!isfinite(maximum) || minimum == DBL_MAX || minimum == 0.0) {
        return NAN;
    }
    return maximum / minimum;
}

static void initialize_gaussian_result(
    GaussianResult *result, const double *initial)
{
    size_t i;

    for (i = 0; i < GAUSSIAN_PARAMETER_COUNT; ++i) {
        result->params[i] = initial[i];
    }
    result->m = 0;
    result->m_aug = 0;
    result->outer_callback_count = 0;
    result->status = NLS_ERR_INVALID;
    result->failed = 1;
    result->finite_check = 0;
    result->jac_error = NAN;
    result->initial_violation = HUGE_VAL;
    result->rmse = NAN;
    result->prediction_rmse = NAN;
    result->max_prediction_error = NAN;
    result->relative_prediction_rmse = NAN;
    result->violation = HUGE_VAL;
    result->rho = NAN;
    result->rho_max = NAN;
    result->parameter_scale_ratio = NAN;
    result->theta_jacobian_norm = NAN;
    result->underflow_zero_count = 0;
    for (i = 0; i < GAUSSIAN_MAX_ACTIVE_BOUND_COUNT; ++i) {
        result->active_bound_values[i] = NAN;
        result->active_bound_targets[i] = NAN;
        result->active_bound_slacks[i] = NAN;
        result->active_bound_parameters[i] = 0;
        result->active_bound_kinds[i] = PARAMETER_LOWER_BOUND;
    }
}

static int gaussian_case_valid(const GaussianCase *test_case)
{
    size_t i;

    if (test_case == NULL || test_case->name == NULL ||
        test_case->special_condition == NULL ||
        test_case->reference_expectation == NULL ||
        test_case->truth == NULL || test_case->initial == NULL ||
        test_case->bounds == NULL || test_case->bound_target_text == NULL ||
        test_case->bound_count == 0 ||
        test_case->bound_count > GAUSSIAN_MAX_BOUND_COUNT ||
        test_case->nx < 2 || test_case->ny < 2 ||
        test_case->nx > SIZE_MAX / test_case->ny ||
        !isfinite(test_case->x_min) || !isfinite(test_case->x_max) ||
        !isfinite(test_case->y_min) || !isfinite(test_case->y_max) ||
        test_case->x_min >= test_case->x_max ||
        test_case->y_min >= test_case->y_max ||
        !isfinite(test_case->constraint_tol) ||
        test_case->constraint_tol < 0.0 ||
        test_case->max_outer_iter == 0 ||
        test_case->expected_active_bound_count >
            GAUSSIAN_MAX_ACTIVE_BOUND_COUNT ||
        (test_case->expected_active_bound_count != 0 &&
         test_case->expected_active_bounds == NULL)) {
        return 0;
    }
    for (i = 0; i < test_case->expected_active_bound_count; ++i) {
        if (test_case->expected_active_bounds[i] >= test_case->bound_count) {
            return 0;
        }
    }
    return finite_values(test_case->truth, GAUSSIAN_PARAMETER_COUNT) &&
           finite_values(test_case->initial, GAUSSIAN_PARAMETER_COUNT);
}

static int execute_gaussian_case(
    const GaussianCase *test_case, GaussianResult *result)
{
    double *xs = NULL;
    double *ys = NULL;
    double *zs = NULL;
    double *residuals = NULL;
    double *jacobian = NULL;
    ParameterBoundData bound_data[GAUSSIAN_MAX_BOUND_COUNT];
    AugLagConstraint inequalities[GAUSSIAN_MAX_BOUND_COUNT];
    AugLagConstraintSet constraints = {NULL, 0, inequalities, 0};
    GaussianFitData fit;
    AugLagContext ctx;
    double residual_sum_squares = 0.0;
    double prediction_sum_squares = 0.0;
    double theta_sum_squares = 0.0;
    double maximum_observed = 0.0;
    size_t ix;
    size_t iy;
    size_t i;
    int preflight_failed = 0;
    int initialized = 0;
    int failed = 0;

    if (result == NULL) {
        return 1;
    }
    if (test_case == NULL || test_case->initial == NULL) {
        return 1;
    }
    initialize_gaussian_result(result, test_case->initial);
    if (!gaussian_case_valid(test_case)) {
        return 1;
    }
    result->m = test_case->nx * test_case->ny;
    if (result->m > SIZE_MAX /
            (GAUSSIAN_PARAMETER_COUNT * sizeof(*jacobian)) ||
        result->m > SIZE_MAX / sizeof(*xs)) {
        return 1;
    }

    xs = (double *)malloc(result->m * sizeof(*xs));
    ys = (double *)malloc(result->m * sizeof(*ys));
    zs = (double *)malloc(result->m * sizeof(*zs));
    residuals = (double *)malloc(result->m * sizeof(*residuals));
    jacobian = (double *)malloc(
        result->m * GAUSSIAN_PARAMETER_COUNT * sizeof(*jacobian));
    if (xs == NULL || ys == NULL || zs == NULL || residuals == NULL ||
        jacobian == NULL) {
        goto cleanup;
    }

    for (iy = 0; iy < test_case->ny; ++iy) {
        double y = test_case->y_min +
                   (test_case->y_max - test_case->y_min) * (double)iy /
                       (double)(test_case->ny - 1);
        for (ix = 0; ix < test_case->nx; ++ix) {
            double x = test_case->x_min +
                       (test_case->x_max - test_case->x_min) * (double)ix /
                           (double)(test_case->nx - 1);
            double noise;
            double exponential;

            i = iy * test_case->nx + ix;
            noise = test_case->noise_scale *
                    sin(0.37 * (double)i) * cos(0.11 * (double)i);
            xs[i] = x;
            ys[i] = y;
            zs[i] = gaussian_model(test_case->truth, x, y) + noise;
            exponential = gaussian_exponential(test_case->truth, x, y);
            if (exponential == 0.0) {
                ++result->underflow_zero_count;
            }
            if (!isfinite(zs[i]) || !isfinite(exponential)) {
                preflight_failed = 1;
            }
            if (fabs(zs[i]) > maximum_observed) {
                maximum_observed = fabs(zs[i]);
            }
        }
    }

    fit.count = result->m;
    fit.x = xs;
    fit.y = ys;
    fit.z = zs;
    fit.outer_params = NULL;
    fit.outer_iterations = NULL;

    if (gaussian_jacobian_fd_check(
            &fit,
            test_case->initial,
            test_case->narrow_fd_steps,
            &result->jac_error) != 0 ||
        !isfinite(result->jac_error) ||
        result->jac_error >= (test_case->max_jac_error > 0.0
                                  ? test_case->max_jac_error
                                  : 1e-5)) {
        preflight_failed = 1;
    }

    constraints.nineq = test_case->bound_count;
    for (i = 0; i < test_case->bound_count; ++i) {
        bound_data[i] = test_case->bounds[i];
        inequalities[i].eval = parameter_bound_eval;
        inequalities[i].jac = parameter_bound_jac;
        inequalities[i].data = &bound_data[i];
        inequalities[i].tol = test_case->constraint_tol;
    }
    result->initial_violation = constraint_violation(
        &constraints, test_case->initial);

    if (!preflight_failed) {
        result->status = auglag_init(&ctx,
                                     gaussian_residual,
                                     gaussian_jacobian,
                                     &fit,
                                     result->m,
                                     GAUSSIAN_PARAMETER_COUNT,
                                     &constraints,
                                     NLS_ALGO_LM,
                                     LLS_ALGO_CHOLESKY);
        if (result->status == 0) {
            initialized = 1;
            ctx.constraint_tol = test_case->constraint_tol;
            ctx.max_outer_iter = test_case->max_outer_iter;
            ctx.rho = 1.0;
            ctx.rho_factor = 10.0;
            ctx.rho_update_tau = 0.5;
            ctx.rho_max = 1e12;
            result->m_aug = ctx.aug_m;
            fit.outer_params = result->params;
            fit.outer_iterations = &result->outer_callback_count;
            result->status = auglag_solve(&ctx, result->params);
            fit.outer_params = NULL;
            fit.outer_iterations = NULL;
            result->violation = constraint_violation(
                &constraints, result->params);
            result->rho = ctx.rho;
            result->rho_max = ctx.rho_max;
        }
    }

    result->finite_check =
        finite_values(result->params, GAUSSIAN_PARAMETER_COUNT) &&
        gaussian_residual(&fit,
                          result->m,
                          GAUSSIAN_PARAMETER_COUNT,
                          result->params,
                          residuals) == 0 &&
        gaussian_jacobian(&fit,
                          result->m,
                          GAUSSIAN_PARAMETER_COUNT,
                          result->params,
                          jacobian) == 0 &&
        finite_values(residuals, result->m) &&
        finite_values(
            jacobian, result->m * GAUSSIAN_PARAMETER_COUNT);
    if (result->finite_check) {
        result->max_prediction_error = 0.0;
        for (i = 0; i < result->m; ++i) {
            double fitted = gaussian_model(result->params, xs[i], ys[i]);
            double expected = gaussian_model(test_case->truth, xs[i], ys[i]);
            double prediction_difference = fitted - expected;
            double prediction_error = fabs(prediction_difference);
            double theta_value;

            residual_sum_squares += residuals[i] * residuals[i];
            prediction_sum_squares +=
                prediction_difference * prediction_difference;
            if (prediction_error > result->max_prediction_error) {
                result->max_prediction_error = prediction_error;
            }
            theta_value = test_case->truth[1] *
                gaussian_exponential(test_case->truth, xs[i], ys[i]);
            {
                double dx = xs[i] - test_case->truth[5];
                double dy = ys[i] - test_case->truth[6];
                double ct = cos(test_case->truth[4]);
                double st = sin(test_case->truth[4]);
                double u = ct * dx + st * dy;
                double v = -st * dx + ct * dy;
                double inv_sx2 = 1.0 /
                    (test_case->truth[2] * test_case->truth[2]);
                double inv_sy2 = 1.0 /
                    (test_case->truth[3] * test_case->truth[3]);

                theta_value *= u * v * (inv_sy2 - inv_sx2);
            }
            theta_sum_squares += theta_value * theta_value;
        }
        result->rmse = sqrt(residual_sum_squares / (double)result->m);
        result->prediction_rmse =
            sqrt(prediction_sum_squares / (double)result->m);
        result->relative_prediction_rmse = maximum_observed > 0.0
            ? result->prediction_rmse / maximum_observed
            : NAN;
        result->theta_jacobian_norm = sqrt(theta_sum_squares);
        result->parameter_scale_ratio =
            gaussian_parameter_scale_ratio(result->params);
        result->finite_check &= isfinite(result->rmse) &&
            isfinite(result->prediction_rmse) &&
            isfinite(result->max_prediction_error) &&
            isfinite(result->relative_prediction_rmse) &&
            isfinite(result->theta_jacobian_norm) &&
            isfinite(result->parameter_scale_ratio);
    }

    failed |= preflight_failed;
    failed |= result->m_aug != result->m + test_case->bound_count;
    failed |= result->outer_callback_count == 0;
    if (test_case->expected_failure) {
        failed |= result->status == AUGLAG_SUCCESS;
        failed |= !result->finite_check;
        failed |= !isfinite(result->violation) ||
                  result->violation <= test_case->constraint_tol;
        failed |= !isfinite(result->rho) || !isfinite(result->rho_max) ||
                  result->rho > result->rho_max;
    } else {
        failed |= result->status != AUGLAG_SUCCESS;
        failed |= !result->finite_check;
        failed |= !isfinite(result->violation) ||
                  result->violation > test_case->constraint_tol;
        failed |= !isfinite(result->rmse) ||
                  result->rmse >= test_case->max_rmse;
        failed |= !isfinite(result->max_prediction_error) ||
                  result->max_prediction_error >=
                      test_case->max_prediction_error;
        if (isfinite(test_case->max_relative_prediction_rmse)) {
            failed |= !isfinite(result->relative_prediction_rmse) ||
                      result->relative_prediction_rmse >=
                          test_case->max_relative_prediction_rmse;
        }
        for (i = 0; i < test_case->bound_count; ++i) {
            failed |= parameter_bound_slack(&bound_data[i], result->params) <
                      -test_case->constraint_tol;
        }
    }
    if (test_case->require_initial_infeasible) {
        failed |= !isfinite(result->initial_violation) ||
                  result->initial_violation <= test_case->constraint_tol;
    } else {
        failed |= !isfinite(result->initial_violation) ||
                  result->initial_violation > test_case->constraint_tol;
    }
    for (i = 0; i < test_case->expected_active_bound_count; ++i) {
        const ParameterBoundData *active =
            &bound_data[test_case->expected_active_bounds[i]];

        result->active_bound_parameters[i] = active->index;
        result->active_bound_kinds[i] = active->kind;
        result->active_bound_values[i] = result->params[active->index];
        result->active_bound_targets[i] = active->target;
        result->active_bound_slacks[i] =
            parameter_bound_slack(active, result->params);
        if (!test_case->expected_failure) {
            failed |= fabs(result->active_bound_slacks[i]) >= 1e-4;
        }
    }
    if (test_case->require_box_interior) {
        int matching_bounds = 0;

        for (i = 0; i < test_case->bound_count; ++i) {
            if (bound_data[i].index == 2 &&
                (bound_data[i].target == 1.0 ||
                 bound_data[i].target == 1.4)) {
                ++matching_bounds;
                failed |= parameter_bound_slack(
                              &bound_data[i], result->params) <=
                          1e-3;
            }
        }
        failed |= matching_bounds != 2;
    }
    if (test_case->require_truth_recovery) {
        failed |= fabs(result->params[0] - test_case->truth[0]) >= 1e-4;
        failed |= fabs(result->params[1] - test_case->truth[1]) >= 1e-4;
        failed |= fabs(result->params[5] - test_case->truth[5]) >= 1e-4;
        failed |= fabs(result->params[6] - test_case->truth[6]) >= 1e-4;
    }
    if (test_case->require_equal_widths) {
        failed |= fabs(result->params[2] - result->params[3]) >= 1e-4;
    }
    result->failed = failed;

cleanup:
    if (initialized) {
        auglag_destroy(&ctx);
    }
    free(xs);
    free(ys);
    free(zs);
    free(residuals);
    free(jacobian);
    return result->failed;
}

static void print_double_list(const double *values, size_t count)
{
    size_t i;

    putchar('[');
    for (i = 0; i < count; ++i) {
        printf("%s%.12g", i == 0 ? "" : ", ", values[i]);
    }
    putchar(']');
}

static void print_size_list(const size_t *values, size_t count)
{
    size_t i;

    putchar('[');
    for (i = 0; i < count; ++i) {
        printf("%s%zu", i == 0 ? "" : ", ", values[i]);
    }
    putchar(']');
}

static void print_kind_list(
    const ParameterBoundKind *values, size_t count)
{
    size_t i;

    putchar('[');
    for (i = 0; i < count; ++i) {
        printf("%s%d",
               i == 0 ? "" : ", ",
               values[i] == PARAMETER_LOWER_BOUND ? -1 : 1);
    }
    putchar(']');
}

static void print_gaussian_result(
    const GaussianCase *test_case,
    const GaussianResult *result,
    const GaussianResult *strict_result)
{
    size_t active_count = test_case->expected_active_bound_count;
    const char *active_type = "none";

    if (active_count != 0) {
        active_type = result->active_bound_kinds[0] ==
                              PARAMETER_LOWER_BOUND
                          ? "lower"
                          : "upper";
    }

    if (test_case->print_human) {
        puts("Gaussian large:");
        printf("m=%zu n=%d m_aug=%zu\n",
               result->m,
               GAUSSIAN_PARAMETER_COUNT,
               result->m_aug);
        print_gaussian_params("initial", test_case->initial);
        print_gaussian_params("truth", test_case->truth);
        print_gaussian_params("final", result->params);
        printf("jac_error=%.6g\n", result->jac_error);
        printf("outer=%zu\n", result->outer_callback_count);
        printf("outer_callback_count=%zu\n",
               result->outer_callback_count);
        printf("rmse=%.6g\n", result->rmse);
        printf("max_prediction_error=%.6g\n",
               result->max_prediction_error);
        printf("violation=%.6g\n", result->violation);
        printf("status=%d\n", result->status);
        puts(result->failed ? "FAIL" : "PASS");
        putchar('\n');
    }

    printf("C CASE %s\n", test_case->name);
    printf("m=%zu\n", result->m);
    printf("n=%d\n", GAUSSIAN_PARAMETER_COUNT);
    printf("m_aug=%zu\n", result->m_aug);
    printf("grid_nx=%zu\n", test_case->nx);
    printf("grid_ny=%zu\n", test_case->ny);
    printf("x_min=%.12g\n", test_case->x_min);
    printf("x_max=%.12g\n", test_case->x_max);
    printf("y_min=%.12g\n", test_case->y_min);
    printf("y_max=%.12g\n", test_case->y_max);
    printf("special_condition=%s\n", test_case->special_condition);
    printf("reference_expectation=%s\n",
           test_case->reference_expectation);
    printf("expected_failure=%d\n", test_case->expected_failure);
    printf("constraint_tol=%.12g\n", test_case->constraint_tol);
    printf("status=%d\n", result->status);
    print_gaussian_params("initial", test_case->initial);
    print_gaussian_params("truth", test_case->truth);
    print_gaussian_params("final", result->params);
    printf("jac_error=%.12g\n", result->jac_error);
    puts("jac_error_kind=scaled_relative");
    printf("jac_error_limit=%.12g\n",
           test_case->max_jac_error > 0.0
               ? test_case->max_jac_error
               : 1e-5);
    printf("rmse=%.12g\n", result->rmse);
    printf("prediction_rmse=%.12g\n", result->prediction_rmse);
    printf("relative_prediction_rmse=%.12g\n",
           result->relative_prediction_rmse);
    printf("max_prediction_error=%.12g\n",
           result->max_prediction_error);
    printf("constraint_violation=%.12g\n", result->violation);
    printf("outer_callback_count=%zu\n",
           result->outer_callback_count);
    printf("finite_check=%d\n", result->finite_check);
    printf("rho=%.12g\n", result->rho);
    printf("rho_max=%.12g\n", result->rho_max);
    printf("parameter_scale_ratio=%.12g\n",
           result->parameter_scale_ratio);
    printf("theta_jacobian_norm=%.12g\n",
           result->theta_jacobian_norm);
    printf("underflow_zero_count=%zu\n",
           result->underflow_zero_count);
    printf("bound_target=%s\n", test_case->bound_target_text);
    printf("active_bound_count=%zu\n", active_count);
    if (active_count != 0) {
        printf("active_bound_index=%zu\n",
               result->active_bound_parameters[0]);
        printf("active_bound_type=%s\n", active_type);
        printf("active_bound_value=%.12g\n",
               result->active_bound_values[0]);
        printf("active_bound_target=%.12g\n",
               result->active_bound_targets[0]);
        printf("active_bound_slack=%.12g\n",
               result->active_bound_slacks[0]);
    } else {
        puts("active_bound_index=-1");
        puts("active_bound_type=none");
        puts("active_bound_value=nan");
        puts("active_bound_target=nan");
        puts("active_bound_slack=nan");
    }
    printf("active_bounds=");
    print_size_list(result->active_bound_parameters, active_count);
    putchar('\n');
    printf("active_bound_indices=");
    print_size_list(result->active_bound_parameters, active_count);
    putchar('\n');
    printf("active_bound_kinds=");
    print_kind_list(result->active_bound_kinds, active_count);
    putchar('\n');
    printf("active_bound_values=");
    print_double_list(result->active_bound_values, active_count);
    putchar('\n');
    printf("active_bound_targets=");
    print_double_list(result->active_bound_targets, active_count);
    putchar('\n');
    printf("active_bound_slacks=");
    print_double_list(result->active_bound_slacks, active_count);
    putchar('\n');
    printf("initial_constraint_violation=%.12g\n",
           result->initial_violation);
    if (strict_result != NULL) {
        printf("bound_slack=%.12g\n", result->params[2] - 0.9);
        puts("tol_1e8_constraint_tol=1e-8");
        printf("tol_1e8_status=%d\n", result->status);
        print_gaussian_params("tol_1e8_final", result->params);
        printf("tol_1e8_constraint_violation=%.12g\n",
               result->violation);
        printf("tol_1e8_bound_slack=%.12g\n",
               result->params[2] - 0.9);
        printf("strict_constraint_tol=1e-10\n");
        printf("strict_status=%d\n", strict_result->status);
        print_gaussian_params("strict_final", strict_result->params);
        printf("strict_constraint_violation=%.12g\n",
               strict_result->violation);
        printf("strict_bound_slack=%.12g\n",
               strict_result->params[2] - 0.9);
        printf("strict_prediction_rmse=%.12g\n",
               strict_result->prediction_rmse);
        printf("strict_relative_prediction_rmse=%.12g\n",
               strict_result->relative_prediction_rmse);
        printf("strict_finite_check=%d\n",
               strict_result->finite_check);
        printf("active_by_tolerance_1e8=%d\n",
               result->params[2] - 0.9 <= 1.000001e-8);
        printf("active_by_tolerance_1e10=%d\n",
               strict_result->params[2] - 0.9 <= 1.000001e-10);
        printf("mathematically_interior=%d\n",
               strict_result->params[2] > 0.9);
    }
    printf("test_result=%s\n", result->failed ? "FAIL" : "PASS");
    printf("pass=%d\n\n", result->failed ? 0 : 1);
}

static int run_gaussian_case(const GaussianCase *test_case)
{
    GaussianResult result;

    execute_gaussian_case(test_case, &result);
    print_gaussian_result(test_case, &result, NULL);

    return result.failed;
}

static const double gaussian_truth[GAUSSIAN_PARAMETER_COUNT] = {
    0.30, 2.50, 1.20, 2.00, 0.35, 0.40, -0.60
};

static const double gaussian_initial[GAUSSIAN_PARAMETER_COUNT] = {
    0.10, 1.70, 1.60, 1.50, 0.20, -0.20, 0.10
};

static const ParameterBoundData gaussian_positive_bounds[] = {
    {1, 1e-10, PARAMETER_LOWER_BOUND},
    {2, 1e-10, PARAMETER_LOWER_BOUND},
    {3, 1e-10, PARAMETER_LOWER_BOUND},
    {4, 1e-10, PARAMETER_LOWER_BOUND},
};

static const size_t gaussian_active_lower_bound[] = {1};
static const size_t gaussian_active_upper_bound[] = {4};

static int test_large_gaussian_fit(void)
{
    static const GaussianCase test_case = {
        .name = "gaussian_normal",
        .special_condition = "normal",
        .reference_expectation = "scipy_trf_success",
        .truth = gaussian_truth,
        .initial = gaussian_initial,
        .bounds = gaussian_positive_bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "[1e-10, 1e-10, 1e-10, 1e-10]",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 1e-6,
        .max_prediction_error = 1e-5,
        .max_relative_prediction_rmse = INFINITY,
        .require_truth_recovery = 1,
        .print_human = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_infeasible_initial(void)
{
    static const double initial[GAUSSIAN_PARAMETER_COUNT] = {
        0.10, 1.70, 0.20, 1.50, 0.20, -0.20, 0.10
    };
    static const ParameterBoundData bounds[] = {
        {1, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 0.50, PARAMETER_LOWER_BOUND},
        {3, 1e-10, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
    };
    static const GaussianCase test_case = {
        .name = "gaussian_infeasible_initial",
        .special_condition = "infeasible_initial",
        .reference_expectation = "scipy_trf_projected_initial",
        .truth = gaussian_truth,
        .initial = initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "0.5",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 1e-5,
        .max_prediction_error = 1e-4,
        .max_relative_prediction_rmse = INFINITY,
        .require_initial_infeasible = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_active_lower(void)
{
    static const double truth[GAUSSIAN_PARAMETER_COUNT] = {
        0.30, 2.50, 0.60, 2.00, 0.35, 0.40, -0.60
    };
    static const ParameterBoundData bounds[] = {
        {1, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 0.90, PARAMETER_LOWER_BOUND},
        {3, 1e-10, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
    };
    static const GaussianCase test_case = {
        .name = "gaussian_active_lower",
        .special_condition = "active_lower_bound",
        .reference_expectation = "scipy_trf_success",
        .truth = truth,
        .initial = gaussian_initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "0.9",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 0.20,
        .max_prediction_error = 1.0,
        .max_relative_prediction_rmse = INFINITY,
        .expected_active_bounds = gaussian_active_lower_bound,
        .expected_active_bound_count = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_active_upper(void)
{
    static const ParameterBoundData bounds[] = {
        {1, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 1e-10, PARAMETER_LOWER_BOUND},
        {3, 1e-10, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
        {1, 2.00, PARAMETER_UPPER_BOUND},
    };
    static const GaussianCase test_case = {
        .name = "gaussian_active_upper",
        .special_condition = "active_upper_bound",
        .reference_expectation = "scipy_trf_success",
        .truth = gaussian_truth,
        .initial = gaussian_initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_MAX_BOUND_COUNT,
        .bound_target_text = "2.0",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 0.20,
        .max_prediction_error = 1.0,
        .max_relative_prediction_rmse = INFINITY,
        .expected_active_bounds = gaussian_active_upper_bound,
        .expected_active_bound_count = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_box_bound(void)
{
    static const double initial[GAUSSIAN_PARAMETER_COUNT] = {
        0.10, 1.70, 1.10, 1.50, 0.20, -0.20, 0.10
    };
    static const ParameterBoundData bounds[] = {
        {1, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 1.00, PARAMETER_LOWER_BOUND},
        {3, 1e-10, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 1.40, PARAMETER_UPPER_BOUND},
    };
    static const GaussianCase test_case = {
        .name = "gaussian_box_bound",
        .special_condition = "interior_box_bound",
        .reference_expectation = "scipy_trf_success",
        .truth = gaussian_truth,
        .initial = initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_MAX_BOUND_COUNT,
        .bound_target_text = "[1.0, 1.4]",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 1e-6,
        .max_prediction_error = 1e-5,
        .max_relative_prediction_rmse = INFINITY,
        .require_box_interior = 1,
        .require_truth_recovery = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_noisy(void)
{
    static const GaussianCase test_case = {
        .name = "gaussian_noisy",
        .special_condition = "deterministic_noise",
        .reference_expectation = "scipy_trf_success",
        .truth = gaussian_truth,
        .initial = gaussian_initial,
        .bounds = gaussian_positive_bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "[1e-10, 1e-10, 1e-10, 1e-10]",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .noise_scale = 0.01,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 0.02,
        .max_prediction_error = 0.02,
        .max_relative_prediction_rmse = INFINITY,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_poor_initial(void)
{
    static const double initial[GAUSSIAN_PARAMETER_COUNT] = {
        0.60, 1.20, 2.00, 1.20, 0.60, -0.70, 0.30
    };
    static const GaussianCase test_case = {
        .name = "gaussian_poor_initial",
        .special_condition = "poor_initial_guess",
        .reference_expectation = "scipy_trf_success",
        .truth = gaussian_truth,
        .initial = initial,
        .bounds = gaussian_positive_bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "[1e-10, 1e-10, 1e-10, 1e-10]",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 1e-5,
        .max_prediction_error = 1e-4,
        .max_relative_prediction_rmse = INFINITY,
        .require_truth_recovery = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_bad_scaling(void)
{
    static const double truth[GAUSSIAN_PARAMETER_COUNT] = {
        1e3, 1e6, 1e-2, 2.0, 0.30, 0.01, -0.50
    };
    static const double initial[GAUSSIAN_PARAMETER_COUNT] = {
        900.0, 9e5, 0.01, 2.0, 0.30, 0.01, -0.50
    };
    static const GaussianCase test_case = {
        .name = "gaussian_bad_scaling",
        .special_condition = "severe_parameter_scaling",
        .reference_expectation = "scipy_trf_success",
        .truth = truth,
        .initial = initial,
        .bounds = gaussian_positive_bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "[1e-10, 1e-10, 1e-10, 1e-10]",
        .nx = 101, .ny = 51,
        .x_min = -0.05, .x_max = 0.07,
        .y_min = -4.0, .y_max = 3.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 10.0,
        .max_prediction_error = 100.0,
        .max_relative_prediction_rmse = 1e-6,
        .max_jac_error = 1e-4,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_near_lower_bound(void)
{
    static const double truth[GAUSSIAN_PARAMETER_COUNT] = {
        0.30, 2.50, 0.90000001, 2.00, 0.35, 0.40, -0.60
    };
    static const ParameterBoundData bounds[] = {
        {1, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 0.90, PARAMETER_LOWER_BOUND},
        {3, 1e-10, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
    };
    static const GaussianCase test_case = {
        .name = "gaussian_near_lower_bound",
        .special_condition = "interior_1e-8_from_lower_bound",
        .reference_expectation = "scipy_trf_success",
        .truth = truth,
        .initial = gaussian_initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "0.9",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 1e-6,
        .max_prediction_error = 1e-5,
        .max_relative_prediction_rmse = INFINITY,
    };
    GaussianCase strict_case = test_case;
    GaussianResult result;
    GaussianResult strict_result;

    strict_case.constraint_tol = 1e-10;
    execute_gaussian_case(&test_case, &result);
    execute_gaussian_case(&strict_case, &strict_result);
    result.failed |= strict_result.failed;
    print_gaussian_result(&test_case, &result, &strict_result);
    return result.failed;
}

static int test_gaussian_two_active_bounds(void)
{
    static const double truth[GAUSSIAN_PARAMETER_COUNT] = {
        0.30, 2.50, 0.60, 2.00, 0.35, 0.40, -0.60
    };
    static const ParameterBoundData bounds[] = {
        {1, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 0.90, PARAMETER_LOWER_BOUND},
        {3, 1e-10, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
        {1, 2.00, PARAMETER_UPPER_BOUND},
    };
    static const size_t active_bounds[] = {4, 1};
    static const GaussianCase test_case = {
        .name = "gaussian_two_active_bounds",
        .special_condition = "two_simultaneously_active_bounds",
        .reference_expectation = "scipy_trf_success",
        .truth = truth,
        .initial = gaussian_initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_MAX_BOUND_COUNT,
        .bound_target_text = "[2.0, 0.9]",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 40,
        .max_rmse = 0.30,
        .max_prediction_error = 1.0,
        .max_relative_prediction_rmse = INFINITY,
        .expected_active_bounds = active_bounds,
        .expected_active_bound_count = 2,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_infeasible_box(void)
{
    static const ParameterBoundData bounds[] = {
        {1, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 2.00, PARAMETER_LOWER_BOUND},
        {3, 1e-10, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 1.00, PARAMETER_UPPER_BOUND},
    };
    static const GaussianCase test_case = {
        .name = "gaussian_infeasible_box",
        .special_condition = "contradictory_box_bounds",
        .reference_expectation = "rejected_bounds",
        .truth = gaussian_truth,
        .initial = gaussian_initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_MAX_BOUND_COUNT,
        .bound_target_text = "[2.0, 1.0]",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 12,
        .max_rmse = INFINITY,
        .max_prediction_error = INFINITY,
        .max_relative_prediction_rmse = INFINITY,
        .require_initial_infeasible = 1,
        .expected_failure = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_infeasible_constraints(void)
{
    static const ParameterBoundData bounds[] = {
        {1, 3.00, PARAMETER_LOWER_BOUND},
        {2, 1e-10, PARAMETER_LOWER_BOUND},
        {3, 1e-10, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
        {1, 2.00, PARAMETER_UPPER_BOUND},
    };
    static const GaussianCase test_case = {
        .name = "gaussian_infeasible_constraints",
        .special_condition = "general_infeasible_inequalities",
        .reference_expectation = "infeasible_interval",
        .truth = gaussian_truth,
        .initial = gaussian_initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_MAX_BOUND_COUNT,
        .bound_target_text = "[3.0, 2.0]",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 12,
        .max_rmse = INFINITY,
        .max_prediction_error = INFINITY,
        .max_relative_prediction_rmse = INFINITY,
        .require_initial_infeasible = 1,
        .expected_failure = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_large_grid(void)
{
    static const GaussianCase test_case = {
        .name = "gaussian_large_grid",
        .special_condition = "large_10201_residual_grid",
        .reference_expectation = "scipy_trf_success",
        .truth = gaussian_truth,
        .initial = gaussian_initial,
        .bounds = gaussian_positive_bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "[1e-10, 1e-10, 1e-10, 1e-10]",
        .nx = 101, .ny = 101,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 1e-6,
        .max_prediction_error = 1e-5,
        .max_relative_prediction_rmse = INFINITY,
        .require_truth_recovery = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_rank_deficient(void)
{
    static const double truth[GAUSSIAN_PARAMETER_COUNT] = {
        0.30, 2.50, 1.50, 1.50, 0.35, 0.40, -0.60
    };
    static const double initial[GAUSSIAN_PARAMETER_COUNT] = {
        0.10, 1.70, 1.20, 1.80, 1.10, -0.20, 0.10
    };
    static const GaussianCase test_case = {
        .name = "gaussian_rank_deficient",
        .special_condition = "circular_nonidentifiable_theta",
        .reference_expectation = "scipy_trf_success_nonunique_parameters",
        .truth = truth,
        .initial = initial,
        .bounds = gaussian_positive_bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "[1e-10, 1e-10, 1e-10, 1e-10]",
        .nx = 31, .ny = 31,
        .x_min = -5.0, .x_max = 5.0,
        .y_min = -5.0, .y_max = 5.0,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 1e-6,
        .max_prediction_error = 1e-5,
        .max_relative_prediction_rmse = INFINITY,
        .require_equal_widths = 1,
    };

    return run_gaussian_case(&test_case);
}

static int test_gaussian_narrow_width(void)
{
    static const double truth[GAUSSIAN_PARAMETER_COUNT] = {
        0.10, 2.00, 1e-3, 5e-3, 0.30, 0.0, 0.0
    };
    static const double initial[GAUSSIAN_PARAMETER_COUNT] = {
        0.05, 1.50, 2e-3, 8e-3, 0.20, 1e-3, -2e-3
    };
    static const ParameterBoundData bounds[] = {
        {1, 1e-10, PARAMETER_LOWER_BOUND},
        {2, 1e-5, PARAMETER_LOWER_BOUND},
        {3, 1e-5, PARAMETER_LOWER_BOUND},
        {4, 1e-10, PARAMETER_LOWER_BOUND},
    };
    static const GaussianCase test_case = {
        .name = "gaussian_narrow_width",
        .special_condition = "narrow_width_exponential_underflow",
        .reference_expectation = "scipy_trf_success",
        .truth = truth,
        .initial = initial,
        .bounds = bounds,
        .bound_count = GAUSSIAN_POSITIVE_BOUND_COUNT,
        .bound_target_text = "[1e-10, 1e-5, 1e-5, 1e-10]",
        .nx = 101, .ny = 101,
        .x_min = -0.04, .x_max = 0.04,
        .y_min = -0.12, .y_max = 0.12,
        .constraint_tol = 1e-8,
        .max_outer_iter = 30,
        .max_rmse = 1e-5,
        .max_prediction_error = 1e-4,
        .max_relative_prediction_rmse = 1e-6,
        .narrow_fd_steps = 1,
    };

    return run_gaussian_case(&test_case);
}

int main(void)
{
    int failed = 0;

    failed |= test_unconstrained();
    failed |= test_equality();
    failed |= test_inequality();
    failed |= test_mixed_constraints();
    failed |= test_large_gaussian_fit();
    failed |= test_gaussian_infeasible_initial();
    failed |= test_gaussian_active_lower();
    failed |= test_gaussian_active_upper();
    failed |= test_gaussian_box_bound();
    failed |= test_gaussian_noisy();
    failed |= test_gaussian_poor_initial();
    failed |= test_gaussian_bad_scaling();
    failed |= test_gaussian_near_lower_bound();
    failed |= test_gaussian_two_active_bounds();
    failed |= test_gaussian_infeasible_box();
    failed |= test_gaussian_infeasible_constraints();
    failed |= test_gaussian_large_grid();
    failed |= test_gaussian_rank_deficient();
    failed |= test_gaussian_narrow_width();

    if (failed) {
        fprintf(stderr, "test_auglag failed\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
