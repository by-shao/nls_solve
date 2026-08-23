#include "auglag_constrain.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    GAUSSIAN_NX = 31,
    GAUSSIAN_NY = 31,
    GAUSSIAN_COUNT = GAUSSIAN_NX * GAUSSIAN_NY,
    GAUSSIAN_PARAMETER_COUNT = 7,
    GAUSSIAN_BOUND_COUNT = 4
};

typedef struct {
    size_t count;
    const double *x;
    const double *y;
    const double *z;
    const double *outer_params;
    size_t *outer_iterations;
} GaussianFitData;

typedef struct {
    size_t index;
    double lower;
} LowerBoundData;

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

static int lower_bound_eval(const double *x, double *value, void *data)
{
    const LowerBoundData *bound = (const LowerBoundData *)data;

    if (x == NULL || value == NULL || bound == NULL ||
        bound->index >= GAUSSIAN_PARAMETER_COUNT) {
        return 1;
    }
    *value = bound->lower - x[bound->index];
    return 0;
}

static int lower_bound_jac(const double *x, double *jac, void *data)
{
    const LowerBoundData *bound = (const LowerBoundData *)data;
    size_t j;

    if (x == NULL || jac == NULL || bound == NULL ||
        bound->index >= GAUSSIAN_PARAMETER_COUNT) {
        return 1;
    }
    for (j = 0; j < GAUSSIAN_PARAMETER_COUNT; ++j) {
        jac[j] = 0.0;
    }
    jac[bound->index] = -1.0;
    return 0;
}

static int gaussian_jacobian_fd_check(
    GaussianFitData *fit, const double *params, double *max_error)
{
    const double eps = 1e-6;
    double analytic[GAUSSIAN_COUNT * GAUSSIAN_PARAMETER_COUNT];
    double fplus[GAUSSIAN_COUNT];
    double fminus[GAUSSIAN_COUNT];
    double plus[GAUSSIAN_PARAMETER_COUNT];
    double minus[GAUSSIAN_PARAMETER_COUNT];
    size_t i;
    size_t j;

    if (fit == NULL || params == NULL || max_error == NULL ||
        gaussian_jacobian(fit,
                          fit->count,
                          GAUSSIAN_PARAMETER_COUNT,
                          params,
                          analytic) != 0) {
        return 1;
    }

    *max_error = 0.0;
    for (j = 0; j < GAUSSIAN_PARAMETER_COUNT; ++j) {
        for (i = 0; i < GAUSSIAN_PARAMETER_COUNT; ++i) {
            plus[i] = params[i];
            minus[i] = params[i];
        }
        plus[j] += eps;
        minus[j] -= eps;
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
            return 1;
        }
        for (i = 0; i < fit->count; ++i) {
            double finite_difference = (fplus[i] - fminus[i]) / (2.0 * eps);
            double error =
                fabs(analytic[i * GAUSSIAN_PARAMETER_COUNT + j] -
                     finite_difference);
            if (error > *max_error) {
                *max_error = error;
            }
        }
    }
    return 0;
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

static int test_large_gaussian_fit(void)
{
    static const double truth[GAUSSIAN_PARAMETER_COUNT] = {
        0.30, 2.50, 1.20, 2.00, 0.35, 0.40, -0.60
    };
    static const double initial[GAUSSIAN_PARAMETER_COUNT] = {
        0.10, 1.70, 1.60, 1.50, 0.20, -0.20, 0.10
    };
    double xs[GAUSSIAN_COUNT];
    double ys[GAUSSIAN_COUNT];
    double zs[GAUSSIAN_COUNT];
    double residuals[GAUSSIAN_COUNT];
    double params[GAUSSIAN_PARAMETER_COUNT];
    LowerBoundData bound_data[GAUSSIAN_BOUND_COUNT];
    AugLagConstraint inequalities[GAUSSIAN_BOUND_COUNT];
    AugLagConstraintSet constraints = {
        NULL, 0, inequalities, GAUSSIAN_BOUND_COUNT
    };
    GaussianFitData fit;
    AugLagContext ctx;
    size_t outer_iterations = 0;
    size_t m_aug = 0;
    double jac_error = NAN;
    double rmse = NAN;
    double max_prediction_error = NAN;
    double violation = HUGE_VAL;
    double sum_squares = 0.0;
    size_t ix;
    size_t iy;
    size_t i;
    size_t j;
    int status = NLS_ERR_INVALID;
    int failed = 0;

    for (j = 0; j < GAUSSIAN_PARAMETER_COUNT; ++j) {
        params[j] = initial[j];
    }
    for (iy = 0; iy < GAUSSIAN_NY; ++iy) {
        double y = -5.0 + 10.0 * (double)iy / (double)(GAUSSIAN_NY - 1);
        for (ix = 0; ix < GAUSSIAN_NX; ++ix) {
            double x =
                -5.0 + 10.0 * (double)ix / (double)(GAUSSIAN_NX - 1);
            i = iy * GAUSSIAN_NX + ix;
            xs[i] = x;
            ys[i] = y;
            zs[i] = gaussian_model(truth, x, y);
        }
    }

    fit.count = GAUSSIAN_COUNT;
    fit.x = xs;
    fit.y = ys;
    fit.z = zs;
    fit.outer_params = NULL;
    fit.outer_iterations = NULL;

    if (gaussian_jacobian_fd_check(&fit, initial, &jac_error) != 0 ||
        !isfinite(jac_error) || jac_error >= 1e-5) {
        failed = 1;
    }

    for (i = 0; i < GAUSSIAN_BOUND_COUNT; ++i) {
        bound_data[i].index = i + 1;
        bound_data[i].lower = 1e-10;
        inequalities[i].eval = lower_bound_eval;
        inequalities[i].jac = lower_bound_jac;
        inequalities[i].data = &bound_data[i];
        inequalities[i].tol = 1e-8;
    }

    if (!failed) {
        status = auglag_init(&ctx,
                             gaussian_residual,
                             gaussian_jacobian,
                             &fit,
                             GAUSSIAN_COUNT,
                             GAUSSIAN_PARAMETER_COUNT,
                             &constraints,
                             NLS_ALGO_LM,
                             LLS_ALGO_CHOLESKY);
        if (status == 0) {
            ctx.constraint_tol = 1e-8;
            ctx.max_outer_iter = 20;
            ctx.rho = 1.0;
            ctx.rho_factor = 10.0;
            ctx.rho_update_tau = 0.5;
            ctx.rho_max = 1e12;
            m_aug = ctx.aug_m;
            fit.outer_params = params;
            fit.outer_iterations = &outer_iterations;
            status = auglag_solve(&ctx, params);
            fit.outer_params = NULL;
            fit.outer_iterations = NULL;
            violation = constraint_violation(&constraints, params);
            auglag_destroy(&ctx);
        } else {
            failed = 1;
        }
    }

    if (status == AUGLAG_SUCCESS &&
        gaussian_residual(&fit,
                          GAUSSIAN_COUNT,
                          GAUSSIAN_PARAMETER_COUNT,
                          params,
                          residuals) == 0) {
        for (i = 0; i < GAUSSIAN_COUNT; ++i) {
            double fitted = gaussian_model(params, xs[i], ys[i]);
            double expected = gaussian_model(truth, xs[i], ys[i]);
            double prediction_error = fabs(fitted - expected);

            sum_squares += residuals[i] * residuals[i];
            if (prediction_error > max_prediction_error ||
                !isfinite(max_prediction_error)) {
                max_prediction_error = prediction_error;
            }
        }
        rmse = sqrt(sum_squares / (double)GAUSSIAN_COUNT);
    }

    failed |= status != AUGLAG_SUCCESS;
    failed |= GAUSSIAN_COUNT != 961 ||
              GAUSSIAN_PARAMETER_COUNT != 7 || m_aug != 965;
    failed |= !isfinite(violation) || violation > 1e-8;
    failed |= !isfinite(rmse) || rmse >= 1e-6;
    failed |= !isfinite(max_prediction_error) ||
              max_prediction_error >= 1e-5;
    failed |= outer_iterations == 0 || outer_iterations > 20;
    for (i = 0; i < GAUSSIAN_BOUND_COUNT; ++i) {
        failed |= params[i + 1] < bound_data[i].lower;
    }
    failed |= fabs(params[0] - truth[0]) >= 1e-4;
    failed |= fabs(params[1] - truth[1]) >= 1e-4;
    failed |= fabs(params[5] - truth[5]) >= 1e-4;
    failed |= fabs(params[6] - truth[6]) >= 1e-4;

    puts("Gaussian large:");
    printf("m=%d n=%d m_aug=%zu\n",
           GAUSSIAN_COUNT,
           GAUSSIAN_PARAMETER_COUNT,
           m_aug);
    print_gaussian_params("initial", initial);
    print_gaussian_params("truth", truth);
    print_gaussian_params("final", params);
    printf("jac_error=%.6g\n", jac_error);
    printf("outer=%zu\n", outer_iterations);
    printf("rmse=%.6g\n", rmse);
    printf("max_prediction_error=%.6g\n", max_prediction_error);
    printf("violation=%.6g\n", violation);
    printf("status=%d\n", status);
    puts(failed ? "FAIL" : "PASS");

    return failed;
}

int main(void)
{
    int failed = 0;

    failed |= test_unconstrained();
    failed |= test_equality();
    failed |= test_inequality();
    failed |= test_mixed_constraints();
    failed |= test_large_gaussian_fit();

    if (failed) {
        fprintf(stderr, "test_auglag failed\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
