#include "auglag_constrain.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_COUNT(values) (sizeof(values) / sizeof((values)[0]))

typedef struct {
    size_t count;
    double truth_sum;
} RankDeficientData;

typedef struct {
    size_t count;
    double coordinate_scale;
    double intercept;
    double slope;
} ScaledFitData;

typedef enum {
    OPTION_RHO_INIT,
    OPTION_RHO_FACTOR,
    OPTION_RHO_UPDATE_TAU,
    OPTION_RHO_MAX,
    OPTION_CONSTRAINT_TOL,
    OPTION_MAX_OUTER_ITER
} OptionField;

typedef struct {
    const char *name;
    OptionField field;
    double value;
} InvalidOptionCase;

static const char *algorithm_name(NlsAlgorithm algorithm)
{
    return algorithm == NLS_ALGO_LM ? "LM" : "GN";
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

static int quadratic_residual(const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    (void)data;
    if (m != 2 || n != 2 || x == NULL || fvec == NULL) {
        return 1;
    }
    fvec[0] = x[0] - 2.0;
    fvec[1] = x[1] - 3.0;
    return 0;
}

static int quadratic_jacobian(const void *data, size_t m, size_t n, const double *x, double *jac)
{
    (void)data;
    (void)x;
    if (m != 2 || n != 2 || jac == NULL) {
        return 1;
    }
    jac[0] = 1.0;
    jac[1] = 0.0;
    jac[2] = 0.0;
    jac[3] = 1.0;
    return 0;
}

static int nonfinite_residual(const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    (void)data;
    if (m != 2 || n != 2 || x == NULL || fvec == NULL) {
        return 1;
    }
    fvec[0] = NAN;
    fvec[1] = x[1];
    return 0;
}

static int scalar_residual(const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    (void)data;
    if (m != 1 || n != 1 || x == NULL || fvec == NULL) {
        return 1;
    }
    fvec[0] = x[0];
    return 0;
}

static int scalar_jacobian(const void *data, size_t m, size_t n, const double *x, double *jac)
{
    (void)data;
    (void)x;
    if (m != 1 || n != 1 || jac == NULL) {
        return 1;
    }
    jac[0] = 1.0;
    return 0;
}

static int sum_eq_six(const double *x, double *value, void *data)
{
    (void)data;
    if (x == NULL || value == NULL) {
        return 1;
    }
    *value = x[0] + x[1] - 6.0;
    return 0;
}

static int sum_eq_six_jac(const double *x, double *jac, void *data)
{
    (void)x;
    (void)data;
    if (jac == NULL) {
        return 1;
    }
    jac[0] = 1.0;
    jac[1] = 1.0;
    return 0;
}

static int lower_one(const double *x, double *value, void *data)
{
    (void)data;
    if (x == NULL || value == NULL) {
        return 1;
    }
    *value = 1.0 - x[0];
    return 0;
}

static int lower_one_jac(const double *x, double *jac, void *data)
{
    (void)x;
    (void)data;
    if (jac == NULL) {
        return 1;
    }
    jac[0] = -1.0;
    return 0;
}

static int lower_three_scalar(const double *x, double *value, void *data)
{
    (void)data;
    if (x == NULL || value == NULL) {
        return 1;
    }
    *value = 3.0 - x[0];
    return 0;
}

static int lower_three_scalar_jac(const double *x, double *jac, void *data)
{
    (void)x;
    (void)data;
    if (jac == NULL) {
        return 1;
    }
    jac[0] = -1.0;
    return 0;
}

static int upper_two_scalar(const double *x, double *value, void *data)
{
    (void)data;
    if (x == NULL || value == NULL) {
        return 1;
    }
    *value = x[0] - 2.0;
    return 0;
}

static int upper_two_scalar_jac(const double *x, double *jac, void *data)
{
    (void)x;
    (void)data;
    if (jac == NULL) {
        return 1;
    }
    jac[0] = 1.0;
    return 0;
}

static int lower_three_first(const double *x, double *value, void *data)
{
    (void)data;
    if (x == NULL || value == NULL) {
        return 1;
    }
    *value = 3.0 - x[0];
    return 0;
}

static int lower_three_first_jac(const double *x, double *jac, void *data)
{
    (void)x;
    (void)data;
    if (jac == NULL) {
        return 1;
    }
    jac[0] = -1.0;
    jac[1] = 0.0;
    return 0;
}

static double constraint_violation(const AugLagConstraintSet *constraints, const double *x)
{
    double violation = 0.0;
    size_t i;

    if (constraints == NULL) {
        return 0.0;
    }
    for (i = 0; i < constraints->neq; ++i) {
        double value;

        if (constraints->eq[i].eval(
                x, &value, constraints->eq[i].data) != 0 ||
            !isfinite(value)) {
            return HUGE_VAL;
        }
        violation = fmax(violation, fabs(value));
    }
    for (i = 0; i < constraints->nineq; ++i) {
        double value;

        if (constraints->ineq[i].eval(
                x, &value, constraints->ineq[i].data) != 0 ||
            !isfinite(value)) {
            return HUGE_VAL;
        }
        violation = fmax(violation, fmax(0.0, value));
    }
    return violation;
}

static double rank_coordinate(size_t index, size_t count)
{
    return -2.0 + 4.0 * (double)index / (double)(count - 1);
}

static int rank_deficient_residual(const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    const RankDeficientData *fit = (const RankDeficientData *)data;
    size_t i;

    if (fit == NULL || fit->count != m || m < 2 || n != 2 ||
        x == NULL || fvec == NULL) {
        return 1;
    }
    for (i = 0; i < m; ++i) {
        const double t = rank_coordinate(i, m);
        fvec[i] = (x[0] + x[1] - fit->truth_sum) * t;
    }
    return 0;
}

static int rank_deficient_jacobian(const void *data, size_t m, size_t n, const double *x, double *jac)
{
    const RankDeficientData *fit = (const RankDeficientData *)data;
    size_t i;

    (void)x;
    if (fit == NULL || fit->count != m || m < 2 || n != 2 ||
        jac == NULL) {
        return 1;
    }
    for (i = 0; i < m; ++i) {
        const double t = rank_coordinate(i, m);
        jac[i * n] = t;
        jac[i * n + 1] = t;
    }
    return 0;
}

static double rank_prediction_rmse(const RankDeficientData *fit, const double *x)
{
    double sum_squares = 0.0;
    size_t i;

    for (i = 0; i < fit->count; ++i) {
        const double t = rank_coordinate(i, fit->count);
        const double difference = (x[0] + x[1] - fit->truth_sum) * t;
        sum_squares += difference * difference;
    }
    return sqrt(sum_squares / (double)fit->count);
}

static double scaled_coordinate(const ScaledFitData *fit, size_t index)
{
    return fit->coordinate_scale *
        (-1.0 + 2.0 * (double)index / (double)(fit->count - 1));
}

static int scaled_fit_residual(const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    const ScaledFitData *fit = (const ScaledFitData *)data;
    size_t i;

    if (fit == NULL || fit->count != m || m < 2 || n != 2 ||
        x == NULL || fvec == NULL) {
        return 1;
    }
    for (i = 0; i < m; ++i) {
        const double t = scaled_coordinate(fit, i);
        const double expected = fit->intercept + fit->slope * t;
        fvec[i] = x[0] + x[1] * t - expected;
    }
    return 0;
}

static int scaled_fit_jacobian(const void *data, size_t m, size_t n, const double *x, double *jac)
{
    const ScaledFitData *fit = (const ScaledFitData *)data;
    size_t i;

    (void)x;
    if (fit == NULL || fit->count != m || m < 2 || n != 2 ||
        jac == NULL) {
        return 1;
    }
    for (i = 0; i < m; ++i) {
        jac[i * n] = 1.0;
        jac[i * n + 1] = scaled_coordinate(fit, i);
    }
    return 0;
}

static double scaled_prediction_rmse(const ScaledFitData *fit, const double *x)
{
    double sum_squares = 0.0;
    size_t i;

    for (i = 0; i < fit->count; ++i) {
        const double t = scaled_coordinate(fit, i);
        const double expected = fit->intercept + fit->slope * t;
        const double difference = x[0] + x[1] * t - expected;
        sum_squares += difference * difference;
    }
    return sqrt(sum_squares / (double)fit->count);
}

static int test_core2_mixed(NlsAlgorithm algorithm)
{
    const AugLagProblem problem = {
        quadratic_residual, quadratic_jacobian, NULL, 2, 2
    };
    const AugLagConstraint equalities[] = {
        {sum_eq_six, sum_eq_six_jac, NULL, 1e-8}
    };
    const AugLagConstraint inequalities[] = {
        {lower_three_first, lower_three_first_jac, NULL, 1e-8}
    };
    const AugLagConstraintSet constraints = {
        equalities, ARRAY_COUNT(equalities),
        inequalities, ARRAY_COUNT(inequalities)
    };
    AugLagContext ctx;
    double x[2] = {0.0, 0.0};
    double violation;
    int status;
    int failed;

    status = auglag_init(&ctx, &problem, &constraints, NULL, NULL, algorithm, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("CORE2 MIXED %s: init_status=%d FAIL\n",
               algorithm_name(algorithm), status);
        return 1;
    }

    status = auglag_solve(&ctx, x);
    violation = constraint_violation(&constraints, x);
    failed = status != AUGLAG_SUCCESS || !finite_values(x, 2) ||
        fabs(x[0] - 3.0) >= 1e-6 || fabs(x[1] - 3.0) >= 1e-6 ||
        violation > ctx.options.constraint_tol ||
        !isfinite(ctx.lambda[0]) || !isfinite(ctx.mu[0]) ||
        ctx.mu[0] < 0.0 || !isfinite(ctx.rho) ||
        ctx.rho < ctx.options.rho_init || ctx.rho > ctx.options.rho_max ||
        ctx.options.rho_init != 1.0 || ctx.outer_iterations == 0 ||
        ctx.inner_function_evaluations == 0;

    printf("CORE2 MIXED %s: status=%d x=[%.12g, %.12g] "
           "violation=%.3g lambda=%.6g mu=%.6g rho=%.6g outer=%zu %s\n",
           algorithm_name(algorithm),
           status,
           x[0],
           x[1],
           violation,
           ctx.lambda[0],
           ctx.mu[0],
           ctx.rho,
           ctx.outer_iterations,
           failed ? "FAIL" : "PASS");
    auglag_destroy(&ctx);
    return failed;
}

static int test_core3_active_bound(NlsAlgorithm algorithm)
{
    const AugLagProblem problem = {
        scalar_residual, scalar_jacobian, NULL, 1, 1
    };
    const AugLagConstraint inequalities[] = {
        {lower_one, lower_one_jac, NULL, 1e-8}
    };
    const AugLagConstraintSet constraints = {
        NULL, 0, inequalities, ARRAY_COUNT(inequalities)
    };
    AugLagContext ctx;
    double x[1] = {0.0};
    double violation;
    int status;
    int failed;

    status = auglag_init(&ctx, &problem, &constraints, NULL, NULL, algorithm, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("CORE3 ACTIVE_BOUND %s: init_status=%d FAIL\n",
               algorithm_name(algorithm), status);
        return 1;
    }

    status = auglag_solve(&ctx, x);
    violation = constraint_violation(&constraints, x);
    failed = status != AUGLAG_SUCCESS || !isfinite(x[0]) ||
        fabs(x[0] - 1.0) >= 1e-6 ||
        violation > ctx.options.constraint_tol ||
        !isfinite(ctx.mu[0]) || ctx.mu[0] <= 0.0 ||
        ctx.outer_iterations == 0 || ctx.inner_function_evaluations == 0;

    printf("CORE3 ACTIVE_BOUND %s: status=%d x=%.12g violation=%.3g "
           "mu=%.6g outer=%zu %s\n",
           algorithm_name(algorithm),
           status,
           x[0],
           violation,
           ctx.mu[0],
           ctx.outer_iterations,
           failed ? "FAIL" : "PASS");
    auglag_destroy(&ctx);
    return failed;
}

static int test_core4_infeasible(NlsAlgorithm algorithm)
{
    const AugLagProblem problem = {
        scalar_residual, scalar_jacobian, NULL, 1, 1
    };
    const AugLagConstraint inequalities[] = {
        {lower_three_scalar, lower_three_scalar_jac, NULL, 1e-8},
        {upper_two_scalar, upper_two_scalar_jac, NULL, 1e-8}
    };
    const AugLagConstraintSet constraints = {
        NULL, 0, inequalities, ARRAY_COUNT(inequalities)
    };
    AugLagOptions options;
    AugLagContext ctx;
    double x[1] = {0.0};
    double violation;
    int status;
    int failed;

    auglag_options_init(&options);
    options.max_outer_iter = 8;
    status = auglag_init(&ctx, &problem, &constraints, &options, NULL, algorithm, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("CORE4 INFEASIBLE %s: init_status=%d FAIL\n",
               algorithm_name(algorithm), status);
        return 1;
    }

    status = auglag_solve(&ctx, x);
    violation = constraint_violation(&constraints, x);
    failed = status != AUGLAG_MAX_ITER || !isfinite(x[0]) ||
        !isfinite(violation) || violation <= options.constraint_tol ||
        !isfinite(ctx.final_constraint_violation) ||
        ctx.final_constraint_violation <= options.constraint_tol ||
        !finite_values(ctx.mu, constraints.nineq) ||
        ctx.mu[0] < 0.0 || ctx.mu[1] < 0.0 ||
        ctx.outer_iterations != options.max_outer_iter ||
        !isfinite(ctx.rho) || ctx.rho > options.rho_max;

    printf("CORE4 INFEASIBLE %s: status=%d x=%.12g violation=%.6g "
           "rho=%.6g outer=%zu %s\n",
           algorithm_name(algorithm),
           status,
           x[0],
           violation,
           ctx.rho,
           ctx.outer_iterations,
           failed ? "FAIL" : "PASS");
    auglag_destroy(&ctx);
    return failed;
}

static int test_core5_rank_deficient(void)
{
    RankDeficientData fit = {9, 3.0};
    const AugLagProblem problem = {
        rank_deficient_residual, rank_deficient_jacobian, &fit, 9, 2
    };
    AugLagContext ctx;
    double x[2] = {8.0, -4.0};
    double prediction_rmse;
    int status;
    int failed;

    status = auglag_init(&ctx, &problem, NULL, NULL, NULL, NLS_ALGO_LM, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("CORE5A RANK_DEFICIENT LM: init_status=%d FAIL\n", status);
        return 1;
    }

    status = auglag_solve(&ctx, x);
    prediction_rmse = rank_prediction_rmse(&fit, x);
    failed = status != AUGLAG_SUCCESS || !finite_values(x, 2) ||
        !isfinite(prediction_rmse) || prediction_rmse >= 1e-8 ||
        fabs(x[0] + x[1] - fit.truth_sum) >= 1e-8 ||
        ctx.final_constraint_violation != 0.0 ||
        ctx.outer_iterations == 0 || ctx.inner_function_evaluations == 0;

    printf("CORE5A RANK_DEFICIENT LM: status=%d x=[%.12g, %.12g] "
           "identifiable_sum=%.12g prediction_rmse=%.3g %s\n",
           status,
           x[0],
           x[1],
           x[0] + x[1],
           prediction_rmse,
           failed ? "FAIL" : "PASS");
    puts("CORE5A RANK_DEFICIENT GN: not_applicable="
         "rank-deficient normal equations with Cholesky");
    auglag_destroy(&ctx);
    return failed;
}

static int test_core5_scaled_large_m(NlsAlgorithm algorithm)
{
    ScaledFitData fit = {257, 1.0e3, 2.0, -3.0e-3};
    const AugLagProblem problem = {
        scaled_fit_residual, scaled_fit_jacobian, &fit, 257, 2
    };
    AugLagContext ctx;
    double x[2] = {0.0, 0.0};
    double prediction_rmse;
    int status;
    int failed;

    status = auglag_init(&ctx, &problem, NULL, NULL, NULL, algorithm, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("CORE5B SCALED_LARGE_M %s: init_status=%d FAIL\n",
               algorithm_name(algorithm), status);
        return 1;
    }

    status = auglag_solve(&ctx, x);
    prediction_rmse = scaled_prediction_rmse(&fit, x);
    failed = status != AUGLAG_SUCCESS || !finite_values(x, 2) ||
        fabs(x[0] - fit.intercept) >= 1e-8 ||
        fabs(x[1] - fit.slope) >= 1e-10 ||
        !isfinite(prediction_rmse) || prediction_rmse >= 1e-8 ||
        ctx.aug_m != fit.count || ctx.final_constraint_violation != 0.0 ||
        ctx.outer_iterations == 0 || ctx.inner_function_evaluations == 0;

    printf("CORE5B SCALED_LARGE_M %s: status=%d m=%zu "
           "x=[%.12g, %.12g] prediction_rmse=%.3g %s\n",
           algorithm_name(algorithm),
           status,
           fit.count,
           x[0],
           x[1],
           prediction_rmse,
           failed ? "FAIL" : "PASS");
    auglag_destroy(&ctx);
    return failed;
}

static int options_equal(const AugLagOptions *actual, const AugLagOptions *expected)
{
    return actual->rho_init == expected->rho_init &&
        actual->rho_factor == expected->rho_factor &&
        actual->rho_update_tau == expected->rho_update_tau &&
        actual->rho_max == expected->rho_max &&
        actual->constraint_tol == expected->constraint_tol &&
        actual->max_outer_iter == expected->max_outer_iter;
}

static int expect_invalid_option(const AugLagProblem *problem, const AugLagConstraintSet *constraints, const InvalidOptionCase *test_case)
{
    AugLagOptions options;
    AugLagContext ctx;
    int status;

    auglag_options_init(&options);
    switch (test_case->field) {
    case OPTION_RHO_INIT:
        options.rho_init = test_case->value;
        break;
    case OPTION_RHO_FACTOR:
        options.rho_factor = test_case->value;
        break;
    case OPTION_RHO_UPDATE_TAU:
        options.rho_update_tau = test_case->value;
        break;
    case OPTION_RHO_MAX:
        options.rho_max = test_case->value;
        break;
    case OPTION_CONSTRAINT_TOL:
        options.constraint_tol = test_case->value;
        break;
    case OPTION_MAX_OUTER_ITER:
        options.max_outer_iter = (size_t)test_case->value;
        break;
    }

    status = auglag_init(&ctx, problem, constraints, &options, NULL, NLS_ALGO_LM, LLS_ALGO_CHOLESKY);
    if (status == 0) {
        auglag_destroy(&ctx);
    }
    if (status != NLS_ERR_INVALID) {
        printf("API invalid option %s: status=%d expected=%d FAIL\n",
               test_case->name, status, NLS_ERR_INVALID);
        return 1;
    }
    return 0;
}

static int expect_invalid_multiplier(const AugLagProblem *problem, const AugLagConstraintSet *constraints, const char *name, const AugLagMultiplierInit *multiplier_init)
{
    AugLagContext ctx;
    int status = auglag_init(&ctx, problem, constraints, NULL, multiplier_init, NLS_ALGO_LM, LLS_ALGO_CHOLESKY);

    if (status == 0) {
        auglag_destroy(&ctx);
    }
    if (status != NLS_ERR_INVALID) {
        printf("API invalid multiplier %s: status=%d expected=%d FAIL\n",
               name, status, NLS_ERR_INVALID);
        return 1;
    }
    return 0;
}

static int test_api_matrix(void)
{
    const AugLagProblem problem = {
        quadratic_residual, quadratic_jacobian, NULL, 2, 2
    };
    const AugLagConstraint equalities[] = {
        {sum_eq_six, sum_eq_six_jac, NULL, 1e-8}
    };
    const AugLagConstraint inequalities[] = {
        {lower_three_first, lower_three_first_jac, NULL, 1e-8}
    };
    const AugLagConstraintSet constraints = {
        equalities, ARRAY_COUNT(equalities),
        inequalities, ARRAY_COUNT(inequalities)
    };
    const AugLagProblem numeric_problem = {
        nonfinite_residual, quadratic_jacobian, NULL, 2, 2
    };
    const InvalidOptionCase invalid_options[] = {
        {"rho_init_zero", OPTION_RHO_INIT, 0.0},
        {"rho_init_nan", OPTION_RHO_INIT, NAN},
        {"rho_init_inf", OPTION_RHO_INIT, INFINITY},
        {"rho_factor_one", OPTION_RHO_FACTOR, 1.0},
        {"rho_factor_nan", OPTION_RHO_FACTOR, NAN},
        {"rho_factor_inf", OPTION_RHO_FACTOR, INFINITY},
        {"rho_update_tau_zero", OPTION_RHO_UPDATE_TAU, 0.0},
        {"rho_update_tau_one", OPTION_RHO_UPDATE_TAU, 1.0},
        {"rho_update_tau_nan", OPTION_RHO_UPDATE_TAU, NAN},
        {"rho_update_tau_inf", OPTION_RHO_UPDATE_TAU, INFINITY},
        {"rho_max_below_init", OPTION_RHO_MAX, 0.5},
        {"rho_max_nan", OPTION_RHO_MAX, NAN},
        {"rho_max_inf", OPTION_RHO_MAX, INFINITY},
        {"constraint_tol_zero", OPTION_CONSTRAINT_TOL, 0.0},
        {"constraint_tol_nan", OPTION_CONSTRAINT_TOL, NAN},
        {"constraint_tol_inf", OPTION_CONSTRAINT_TOL, INFINITY},
        {"max_outer_iter_zero", OPTION_MAX_OUTER_ITER, 0.0}
    };
    AugLagOptions defaults;
    AugLagOptions custom;
    AugLagContext ctx;
    AugLagMultiplierInit multiplier_init;
    double lambda_init;
    double mu_init;
    double invalid_value;
    size_t i;
    size_t invalid_option_checks = 0;
    size_t invalid_multiplier_checks = 0;
    int status;
    int failed = 0;

    auglag_options_init(NULL);
    auglag_options_init(&defaults);

    status = auglag_init(&ctx, &problem, &constraints, NULL, NULL, NLS_ALGO_LM, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("API OPT1/MUL1 defaults: init_status=%d FAIL\n", status);
        failed = 1;
    } else {
        const int default_failed =
            !options_equal(&ctx.options, &defaults) ||
            ctx.rho != defaults.rho_init ||
            !isinf(ctx.previous_violation) || ctx.previous_violation < 0.0 ||
            ctx.lambda == NULL || ctx.mu == NULL ||
            ctx.lambda[0] != 0.0 || ctx.mu[0] != 0.0 ||
            ctx.constraint.eq == constraints.eq ||
            ctx.constraint.ineq == constraints.ineq ||
            ctx.problem.residual != problem.residual ||
            ctx.problem.jacobian != problem.jacobian ||
            ctx.problem.m != problem.m || ctx.problem.n != problem.n;

        failed |= default_failed;
        printf("API OPT1/MUL1 defaults: rho=%.6g lambda=%.6g mu=%.6g %s\n",
               ctx.rho,
               ctx.lambda[0],
               ctx.mu[0],
               default_failed ? "FAIL" : "PASS");
        auglag_destroy(&ctx);
    }

    auglag_options_init(&custom);
    custom.rho_init = 2.0;
    custom.rho_factor = 3.0;
    custom.rho_update_tau = 0.25;
    custom.rho_max = 1.0e8;
    custom.constraint_tol = 1.0e-10;
    custom.max_outer_iter = 7;
    lambda_init = -2.5;
    mu_init = 1.25;
    multiplier_init.lambda = &lambda_init;
    multiplier_init.lambda_count = 1;
    multiplier_init.mu = &mu_init;
    multiplier_init.mu_count = 1;
    status = auglag_init(&ctx, &problem, &constraints, &custom, &multiplier_init, NLS_ALGO_LM, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("API OPT2/MUL2 custom-copy: init_status=%d FAIL\n", status);
        failed = 1;
    } else {
        const double saved_lambda = lambda_init;
        const double saved_mu = mu_init;
        int copy_failed = !options_equal(&ctx.options, &custom) ||
            ctx.rho != custom.rho_init ||
            ctx.lambda == &lambda_init || ctx.mu == &mu_init ||
            ctx.lambda[0] != saved_lambda || ctx.mu[0] != saved_mu;

        lambda_init = 101.0;
        mu_init = 202.0;
        copy_failed |= ctx.lambda[0] != saved_lambda ||
            ctx.mu[0] != saved_mu;
        failed |= copy_failed;
        printf("API OPT2/MUL2 custom-copy: rho=%.6g lambda=%.6g mu=%.6g "
               "alias_free=%d %s\n",
               ctx.rho,
               ctx.lambda[0],
               ctx.mu[0],
               copy_failed ? 0 : 1,
               copy_failed ? "FAIL" : "PASS");
        auglag_destroy(&ctx);
    }

    lambda_init = -3.0;
    multiplier_init.lambda = &lambda_init;
    multiplier_init.lambda_count = 1;
    multiplier_init.mu = NULL;
    multiplier_init.mu_count = 0;
    status = auglag_init(&ctx, &problem, &constraints, NULL, &multiplier_init, NLS_ALGO_LM, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("API MUL1 lambda-only: init_status=%d FAIL\n", status);
        failed = 1;
    } else {
        const int partial_failed =
            ctx.lambda[0] != lambda_init || ctx.mu[0] != 0.0;
        failed |= partial_failed;
        printf("API MUL1 lambda-only: lambda=%.6g mu=%.6g %s\n",
               ctx.lambda[0],
               ctx.mu[0],
               partial_failed ? "FAIL" : "PASS");
        auglag_destroy(&ctx);
    }

    mu_init = 0.75;
    multiplier_init.lambda = NULL;
    multiplier_init.lambda_count = 0;
    multiplier_init.mu = &mu_init;
    multiplier_init.mu_count = 1;
    status = auglag_init(&ctx, &problem, &constraints, NULL, &multiplier_init, NLS_ALGO_LM, LLS_ALGO_CHOLESKY);
    if (status != 0) {
        printf("API MUL1 mu-only: init_status=%d FAIL\n", status);
        failed = 1;
    } else {
        const int partial_failed =
            ctx.lambda[0] != 0.0 || ctx.mu[0] != mu_init;
        failed |= partial_failed;
        printf("API MUL1 mu-only: lambda=%.6g mu=%.6g %s\n",
               ctx.lambda[0],
               ctx.mu[0],
               partial_failed ? "FAIL" : "PASS");
        auglag_destroy(&ctx);
    }

    for (i = 0; i < ARRAY_COUNT(invalid_options); ++i) {
        failed |= expect_invalid_option(
            &problem, &constraints, &invalid_options[i]);
        ++invalid_option_checks;
    }

    {
        double numeric_x[2] = {0.0, 0.0};
        int numeric_status = auglag_init(&ctx, &numeric_problem, NULL, NULL, NULL, NLS_ALGO_LM, LLS_ALGO_CHOLESKY);

        if (numeric_status == 0) {
            numeric_status = auglag_solve(&ctx, numeric_x);
            auglag_destroy(&ctx);
        }
        failed |= numeric_status != NLS_ERR_NUMERIC;
        printf("API numeric classification: status=%d expected=%d %s\n",
               numeric_status,
               NLS_ERR_NUMERIC,
               numeric_status == NLS_ERR_NUMERIC ? "PASS" : "FAIL");
    }

    lambda_init = 1.0;
    mu_init = 1.0;
    multiplier_init.lambda = &lambda_init;
    multiplier_init.lambda_count = 0;
    multiplier_init.mu = NULL;
    multiplier_init.mu_count = 0;
    failed |= expect_invalid_multiplier(
        &problem, &constraints, "lambda_count", &multiplier_init);
    ++invalid_multiplier_checks;

    multiplier_init.lambda = NULL;
    multiplier_init.lambda_count = 0;
    multiplier_init.mu = &mu_init;
    multiplier_init.mu_count = 0;
    failed |= expect_invalid_multiplier(
        &problem, &constraints, "mu_count", &multiplier_init);
    ++invalid_multiplier_checks;

    invalid_value = NAN;
    multiplier_init.lambda = &invalid_value;
    multiplier_init.lambda_count = 1;
    multiplier_init.mu = NULL;
    multiplier_init.mu_count = 0;
    failed |= expect_invalid_multiplier(
        &problem, &constraints, "lambda_nan", &multiplier_init);
    ++invalid_multiplier_checks;

    invalid_value = INFINITY;
    failed |= expect_invalid_multiplier(
        &problem, &constraints, "lambda_inf", &multiplier_init);
    ++invalid_multiplier_checks;

    invalid_value = -1.0;
    multiplier_init.lambda = NULL;
    multiplier_init.lambda_count = 0;
    multiplier_init.mu = &invalid_value;
    multiplier_init.mu_count = 1;
    failed |= expect_invalid_multiplier(
        &problem, &constraints, "mu_negative", &multiplier_init);
    ++invalid_multiplier_checks;

    invalid_value = NAN;
    failed |= expect_invalid_multiplier(
        &problem, &constraints, "mu_nan", &multiplier_init);
    ++invalid_multiplier_checks;

    invalid_value = INFINITY;
    failed |= expect_invalid_multiplier(
        &problem, &constraints, "mu_inf", &multiplier_init);
    ++invalid_multiplier_checks;

    printf("API MATRIX: invalid_options=%zu invalid_multipliers=%zu %s\n",
           invalid_option_checks,
           invalid_multiplier_checks,
           failed ? "FAIL" : "PASS");
    return failed;
}

int main(void)
{
    const NlsAlgorithm algorithms[] = {NLS_ALGO_LM, NLS_ALGO_GN};
    size_t i;
    int failed = 0;

    for (i = 0; i < ARRAY_COUNT(algorithms); ++i) {
        failed |= test_core2_mixed(algorithms[i]);
    }
    for (i = 0; i < ARRAY_COUNT(algorithms); ++i) {
        failed |= test_core3_active_bound(algorithms[i]);
    }
    for (i = 0; i < ARRAY_COUNT(algorithms); ++i) {
        failed |= test_core4_infeasible(algorithms[i]);
    }
    failed |= test_core5_rank_deficient();
    for (i = 0; i < ARRAY_COUNT(algorithms); ++i) {
        failed |= test_core5_scaled_large_m(algorithms[i]);
    }
    failed |= test_api_matrix();

    if (failed) {
        fprintf(stderr, "test_auglag failed\n");
        return EXIT_FAILURE;
    }
    puts("test_auglag: PASS (5 numeric cores, 1 API matrix)");
    return EXIT_SUCCESS;
}
