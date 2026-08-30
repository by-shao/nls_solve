#include "auglag_constrain.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DG_PARAMETER_COUNT 7
#define DG_CONSTRAINT_TOL 1.0e-8
#define DG_JACOBIAN_REL_TOL 1.0e-5
#define DG_ACTIVE_BOUND_TOL 2.0e-6

typedef struct {
    const char *name;
    const char *title;
    size_t nx;
    size_t ny;
    double h_min;
    double h_max;
    double v_min;
    double v_max;
    double truth[DG_PARAMETER_COUNT];
    double initial[DG_PARAMETER_COUNT];
    double lower[DG_PARAMETER_COUNT];
    size_t mask_every;
    double noise_scale;
    double reference_params[DG_PARAMETER_COUNT];
    double reference_sse;
    double reference_rmse;
    double reference_max_h;
    double reference_max_v;
    size_t reference_nit;
    size_t reference_nfev;
    double prediction_rel_tol;
    double max_point_tol;
} DoubleGaussianCase;

#include "double_gaussian_reference.h"

typedef struct {
    size_t m;
    double *h;
    double *v;
    double *observed;
} DoubleGaussianData;

typedef struct {
    size_t parameter_index;
    double lower;
} LowerBound;

typedef struct {
    int status;
    double params[DG_PARAMETER_COUNT];
    double sse;
    double rmse;
    double violation;
    double max_h;
    double max_v;
    double rho;
    double prediction_cross_rmse;
    double relative_prediction_cross_rmse;
    double max_point_error;
    double sse_difference;
    double rmse_difference;
    double relative_rmse_difference;
    double parameter_max_abs_difference;
    size_t outer_iterations;
    size_t inner_iterations;
    size_t inner_nfev;
    size_t inner_njev;
    int finite;
} DoubleGaussianResult;

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

static int double_gaussian_model(const double params[DG_PARAMETER_COUNT], double h, double v, double *model)
{
    const double a = params[3] * h + params[4] * v + params[5];
    const double b = params[3] * h - params[4] * v + params[6];
    const double e1 = exp(-a * a);
    const double e2 = exp(-b * b);

    if (!finite_array(params, DG_PARAMETER_COUNT) || !isfinite(h) ||
        !isfinite(v) || !isfinite(a) || !isfinite(b) ||
        !isfinite(e1) || !isfinite(e2)) {
        return 1;
    }
    *model = params[0] + params[1] * e1 + params[2] * e2;
    return isfinite(*model) ? 0 : 1;
}

static int double_gaussian_residual(const void *data, size_t m, size_t n, const double *params, double *fvec)
{
    const DoubleGaussianData *problem = (const DoubleGaussianData *)data;
    size_t i;

    if (problem == NULL || params == NULL || fvec == NULL ||
        m != problem->m || n != DG_PARAMETER_COUNT ||
        !finite_array(params, n)) {
        return 1;
    }
    for (i = 0; i < m; ++i) {
        double model;

        if (double_gaussian_model(
                params, problem->h[i], problem->v[i], &model) != 0 ||
            !isfinite(problem->observed[i])) {
            return 1;
        }
        fvec[i] = model - problem->observed[i];
        if (!isfinite(fvec[i])) {
            return 1;
        }
    }
    return 0;
}

static int double_gaussian_jacobian(const void *data, size_t m, size_t n, const double *params, double *jac)
{
    const DoubleGaussianData *problem = (const DoubleGaussianData *)data;
    size_t i;

    if (problem == NULL || params == NULL || jac == NULL ||
        m != problem->m || n != DG_PARAMETER_COUNT ||
        !finite_array(params, n)) {
        return 1;
    }
    for (i = 0; i < m; ++i) {
        const double h = problem->h[i];
        const double v = problem->v[i];
        const double a = params[3] * h + params[4] * v + params[5];
        const double b = params[3] * h - params[4] * v + params[6];
        const double e1 = exp(-a * a);
        const double e2 = exp(-b * b);
        double *row = jac + i * DG_PARAMETER_COUNT;

        if (!isfinite(h) || !isfinite(v) || !isfinite(a) || !isfinite(b) ||
            !isfinite(e1) || !isfinite(e2)) {
            return 1;
        }
        row[0] = 1.0;
        row[1] = e1;
        row[2] = e2;
        row[3] = -2.0 * params[1] * e1 * a * h -
                 2.0 * params[2] * e2 * b * h;
        row[4] = -2.0 * params[1] * e1 * a * v +
                 2.0 * params[2] * e2 * b * v;
        row[5] = -2.0 * params[1] * e1 * a;
        row[6] = -2.0 * params[2] * e2 * b;
        if (!finite_array(row, DG_PARAMETER_COUNT)) {
            return 1;
        }
    }
    return 0;
}

static int lower_bound_eval(const double *x, double *value, void *data)
{
    const LowerBound *bound = (const LowerBound *)data;

    if (x == NULL || value == NULL || bound == NULL ||
        bound->parameter_index >= DG_PARAMETER_COUNT) {
        return 1;
    }
    *value = bound->lower - x[bound->parameter_index];
    return isfinite(*value) ? 0 : 1;
}

static int lower_bound_jac(const double *x, double *jac, void *data)
{
    const LowerBound *bound = (const LowerBound *)data;

    (void)x;
    if (jac == NULL || bound == NULL ||
        bound->parameter_index >= DG_PARAMETER_COUNT) {
        return 1;
    }
    memset(jac, 0, DG_PARAMETER_COUNT * sizeof(*jac));
    jac[bound->parameter_index] = -1.0;
    return 0;
}

static void free_case_data(DoubleGaussianData *data)
{
    if (data == NULL) {
        return;
    }
    free(data->h);
    free(data->v);
    free(data->observed);
    memset(data, 0, sizeof(*data));
}

static int generate_case_data(const DoubleGaussianCase *test_case, DoubleGaussianData *data)
{
    const size_t full_count = test_case->nx * test_case->ny;
    size_t full_index;
    size_t output_index = 0;

    memset(data, 0, sizeof(*data));
    if (test_case->nx < 2 || test_case->ny < 2 || full_count == 0) {
        return 0;
    }
    for (full_index = 0; full_index < full_count; ++full_index) {
        if (test_case->mask_every == 0 ||
            full_index % test_case->mask_every != 0) {
            ++data->m;
        }
    }
    data->h = (double *)calloc(data->m, sizeof(*data->h));
    data->v = (double *)calloc(data->m, sizeof(*data->v));
    data->observed = (double *)calloc(data->m, sizeof(*data->observed));
    if (data->m == 0 || data->h == NULL || data->v == NULL ||
        data->observed == NULL) {
        free_case_data(data);
        return 0;
    }
    for (full_index = 0; full_index < full_count; ++full_index) {
        const size_t ix = full_index % test_case->nx;
        const size_t iy = full_index / test_case->nx;
        const double h = test_case->h_min + (double)ix *
            (test_case->h_max - test_case->h_min) /
            (double)(test_case->nx - 1);
        const double v = test_case->v_min + (double)iy *
            (test_case->v_max - test_case->v_min) /
            (double)(test_case->ny - 1);
        double observed;

        if (test_case->mask_every != 0 &&
            full_index % test_case->mask_every == 0) {
            continue;
        }
        if (double_gaussian_model(test_case->truth, h, v, &observed) != 0) {
            free_case_data(data);
            return 0;
        }
        observed += test_case->noise_scale *
            (0.01 * sin(0.37 * (double)full_index) +
             0.005 * cos(0.11 * (double)full_index));
        data->h[output_index] = h;
        data->v[output_index] = v;
        data->observed[output_index] = observed;
        ++output_index;
    }
    return output_index == data->m && finite_array(data->observed, data->m);
}

static int jacobian_check(const DoubleGaussianCase *test_case, const DoubleGaussianData *data, double *max_error, double *relative_error)
{
    const size_t mn = data->m * DG_PARAMETER_COUNT;
    double *analytic = (double *)calloc(mn, sizeof(*analytic));
    double *plus = (double *)calloc(data->m, sizeof(*plus));
    double *minus = (double *)calloc(data->m, sizeof(*minus));
    double xp[DG_PARAMETER_COUNT];
    double xm[DG_PARAMETER_COUNT];
    size_t i;
    size_t j;
    int ok = 0;

    if (analytic == NULL || plus == NULL || minus == NULL) {
        goto done;
    }
    if (double_gaussian_jacobian(
            data, data->m, DG_PARAMETER_COUNT, test_case->initial, analytic) != 0) {
        goto done;
    }
    *max_error = 0.0;
    *relative_error = 0.0;
    for (j = 0; j < DG_PARAMETER_COUNT; ++j) {
        const double step = sqrt(DBL_EPSILON) *
            fmax(1.0, fabs(test_case->initial[j]));
        double analytic_sum_squares = 0.0;
        double numeric_sum_squares = 0.0;
        double error_sum_squares = 0.0;

        memcpy(xp, test_case->initial, sizeof(xp));
        memcpy(xm, test_case->initial, sizeof(xm));
        xp[j] += step;
        xm[j] -= step;
        if (double_gaussian_residual(
                data, data->m, DG_PARAMETER_COUNT, xp, plus) != 0 ||
            double_gaussian_residual(
                data, data->m, DG_PARAMETER_COUNT, xm, minus) != 0) {
            goto done;
        }
        for (i = 0; i < data->m; ++i) {
            const double numeric = (plus[i] - minus[i]) / (2.0 * step);
            const double expected = analytic[i * DG_PARAMETER_COUNT + j];
            const double error = fabs(expected - numeric);

            *max_error = fmax(*max_error, error);
            analytic_sum_squares += expected * expected;
            numeric_sum_squares += numeric * numeric;
            error_sum_squares += error * error;
        }
        *relative_error = fmax(
            *relative_error,
            sqrt(error_sum_squares) /
                fmax(
                    fmax(
                        sqrt(analytic_sum_squares),
                        sqrt(numeric_sum_squares)),
                    DBL_MIN));
    }
    ok = isfinite(*max_error) && isfinite(*relative_error);

done:
    free(analytic);
    free(plus);
    free(minus);
    return ok;
}

static double constraint_violation(const DoubleGaussianCase *test_case, const double params[DG_PARAMETER_COUNT])
{
    double violation = 0.0;
    size_t i;

    for (i = 0; i < DG_PARAMETER_COUNT; ++i) {
        if (isfinite(test_case->lower[i])) {
            violation = fmax(violation, test_case->lower[i] - params[i]);
        }
    }
    return fmax(0.0, violation);
}

static int evaluate_result(const DoubleGaussianCase *test_case, const DoubleGaussianData *data, DoubleGaussianResult *result)
{
    double prediction_difference_sum = 0.0;
    double signal_scale = 0.0;
    size_t i;

    result->sse = 0.0;
    result->parameter_max_abs_difference = 0.0;
    for (i = 0; i < data->m; ++i) {
        double prediction;
        double reference_prediction;
        double residual;
        double prediction_difference;

        if (double_gaussian_model(
                result->params, data->h[i], data->v[i], &prediction) != 0 ||
            double_gaussian_model(
                test_case->reference_params,
                data->h[i],
                data->v[i],
                &reference_prediction) != 0) {
            return 0;
        }
        residual = prediction - data->observed[i];
        prediction_difference = prediction - reference_prediction;
        result->sse += residual * residual;
        prediction_difference_sum += prediction_difference * prediction_difference;
        signal_scale = fmax(signal_scale, fabs(data->observed[i]));
    }
    for (i = 0; i < DG_PARAMETER_COUNT; ++i) {
        result->parameter_max_abs_difference = fmax(
            result->parameter_max_abs_difference,
            fabs(result->params[i] - test_case->reference_params[i]));
    }
    result->rmse = sqrt(result->sse / (double)data->m);
    result->violation = constraint_violation(test_case, result->params);
    result->max_h = -(result->params[5] + result->params[6]) /
        (2.0 * result->params[3]);
    result->max_v = -(result->params[5] - result->params[6]) /
        (2.0 * result->params[4]);
    result->prediction_cross_rmse = sqrt(
        prediction_difference_sum / (double)data->m);
    result->relative_prediction_cross_rmse =
        result->prediction_cross_rmse / fmax(signal_scale, DBL_MIN);
    result->max_point_error = hypot(
        result->max_h - test_case->reference_max_h,
        result->max_v - test_case->reference_max_v);
    result->sse_difference = result->sse - test_case->reference_sse;
    result->rmse_difference = result->rmse - test_case->reference_rmse;
    result->relative_rmse_difference =
        fabs(result->rmse_difference) / fmax(signal_scale, DBL_MIN);
    result->finite = finite_array(result->params, DG_PARAMETER_COUNT) &&
        isfinite(result->sse) && isfinite(result->rmse) &&
        isfinite(result->violation) && isfinite(result->max_h) &&
        isfinite(result->max_v) && isfinite(result->rho) &&
        isfinite(result->relative_prediction_cross_rmse) &&
        isfinite(result->max_point_error) &&
        isfinite(result->relative_rmse_difference);
    return result->finite;
}

static void solve_case(const DoubleGaussianCase *test_case, const DoubleGaussianData *data, NlsAlgorithm algorithm, DoubleGaussianResult *result)
{
    AugLagProblem problem = {
        double_gaussian_residual,
        double_gaussian_jacobian,
        (void *)data,
        data->m,
        DG_PARAMETER_COUNT
    };
    LowerBound bound_data[4];
    AugLagConstraint inequalities[4];
    AugLagConstraintSet constraints = {0};
    AugLagOptions options;
    AugLagContext context;
    size_t bound_count = 0;
    size_t i;

    memset(result, 0, sizeof(*result));
    result->rho = NAN;
    memcpy(result->params, test_case->initial, sizeof(result->params));
    for (i = 1; i <= 4; ++i) {
        if (isfinite(test_case->lower[i])) {
            bound_data[bound_count].parameter_index = i;
            bound_data[bound_count].lower = test_case->lower[i];
            inequalities[bound_count].eval = lower_bound_eval;
            inequalities[bound_count].jac = lower_bound_jac;
            inequalities[bound_count].data = &bound_data[bound_count];
            inequalities[bound_count].tol = 0.0;
            ++bound_count;
        }
    }
    constraints.ineq = inequalities;
    constraints.nineq = bound_count;
    auglag_options_init(&options);
    options.max_outer_iter = 50;
    options.constraint_tol = DG_CONSTRAINT_TOL;
    result->status = auglag_init(
        &context,
        &problem,
        &constraints,
        &options,
        NULL,
        algorithm,
        algorithm == NLS_ALGO_LM ? LLS_ALGO_QR : LLS_ALGO_CHOLESKY);
    if (result->status != 0) {
        return;
    }
    result->status = auglag_solve(&context, result->params);
    result->rho = context.rho;
    result->outer_iterations = context.outer_iterations;
    result->inner_iterations = context.inner_iterations;
    result->inner_nfev = context.inner_function_evaluations;
    result->inner_njev = context.inner_jacobian_evaluations;
    (void)evaluate_result(test_case, data, result);
    auglag_destroy(&context);
}

static int result_passes(const DoubleGaussianCase *test_case, const DoubleGaussianResult *result)
{
    const int active_bound_ok = strcmp(test_case->name, "DG5") != 0 ||
        fabs(result->params[3] - test_case->lower[3]) <= DG_ACTIVE_BOUND_TOL;

    return result->status == AUGLAG_SUCCESS && result->finite &&
        result->violation <= DG_CONSTRAINT_TOL &&
        result->relative_prediction_cross_rmse <= test_case->prediction_rel_tol &&
        result->relative_rmse_difference <= test_case->prediction_rel_tol &&
        result->max_point_error <= test_case->max_point_tol &&
        active_bound_ok;
}

static void print_vector(const double *values, size_t count)
{
    size_t i;

    putchar('[');
    for (i = 0; i < count; ++i) {
        printf("%s%.17g", i == 0 ? "" : ",", values[i]);
    }
    putchar(']');
}

static void print_result(const char *algorithm, const DoubleGaussianResult *result, int pass)
{
    printf(
        "  %-3s : status=%d finite=%s sse=%.17g rmse=%.17g "
        "rel_pred=%.6g max_err=%.6g violation=%.6g %s\n",
        algorithm,
        result->status,
        result->finite ? "true" : "false",
        result->sse,
        result->rmse,
        result->relative_prediction_cross_rmse,
        result->max_point_error,
        result->violation,
        pass ? "PASS" : "FAIL");
    printf(
        "        sse_diff=%.6g rmse_diff=%.6g rel_rmse_diff=%.6g "
        "rho=%.6g outer=%zu inner=%zu nfev=%zu njev=%zu\n",
        result->sse_difference,
        result->rmse_difference,
        result->relative_rmse_difference,
        result->rho,
        result->outer_iterations,
        result->inner_iterations,
        result->inner_nfev,
        result->inner_njev);
    printf("        params=");
    print_vector(result->params, DG_PARAMETER_COUNT);
    printf(" param_max_abs_diff=%.6g\n", result->parameter_max_abs_difference);
}

int main(void)
{
    int lm_pass[DOUBLE_GAUSSIAN_CASE_COUNT] = {0};
    int gn_pass[DOUBLE_GAUSSIAN_CASE_COUNT] = {0};
    int overall = 1;
    size_t case_index;

    for (case_index = 0; case_index < DOUBLE_GAUSSIAN_CASE_COUNT; ++case_index) {
        const DoubleGaussianCase *test_case =
            &g_double_gaussian_cases[case_index];
        DoubleGaussianData data;
        DoubleGaussianResult lm_result;
        DoubleGaussianResult gn_result;
        double jac_max_error = NAN;
        double jac_relative_error = NAN;
        int jac_pass;

        printf("%s %s\n", test_case->name, test_case->title);
        printf("  REF : params=");
        print_vector(test_case->reference_params, DG_PARAMETER_COUNT);
        printf(
            " sse=%.17g rmse=%.17g max=[%.17g,%.17g] nit=%zu nfev=%zu\n",
            test_case->reference_sse,
            test_case->reference_rmse,
            test_case->reference_max_h,
            test_case->reference_max_v,
            test_case->reference_nit,
            test_case->reference_nfev);
        if (!generate_case_data(test_case, &data)) {
            printf("  DATA: generation failed\n\n");
            overall = 0;
            continue;
        }
        jac_pass = jacobian_check(
            test_case, &data, &jac_max_error, &jac_relative_error) &&
            jac_relative_error <= DG_JACOBIAN_REL_TOL;
        printf(
            "  JAC : max_abs=%.6g relative=%.6g limit=%.6g %s\n",
            jac_max_error,
            jac_relative_error,
            DG_JACOBIAN_REL_TOL,
            jac_pass ? "PASS" : "FAIL");
        solve_case(test_case, &data, NLS_ALGO_LM, &lm_result);
        solve_case(test_case, &data, NLS_ALGO_GN, &gn_result);
        lm_pass[case_index] = jac_pass && result_passes(test_case, &lm_result);
        gn_pass[case_index] = jac_pass && result_passes(test_case, &gn_result);
        print_result("LM", &lm_result, lm_pass[case_index]);
        print_result("GN", &gn_result, gn_pass[case_index]);
        putchar('\n');
        overall = overall && lm_pass[case_index] && gn_pass[case_index];
        free_case_data(&data);
    }

    printf("Double Gaussian Summary:\n");
    for (case_index = 0; case_index < DOUBLE_GAUSSIAN_CASE_COUNT; ++case_index) {
        printf(
            "  %s: LM %s, GN %s\n",
            g_double_gaussian_cases[case_index].name,
            lm_pass[case_index] ? "PASS" : "FAIL",
            gn_pass[case_index] ? "PASS" : "FAIL");
    }
    printf("  Overall: %s\n", overall ? "PASS" : "FAIL");
    return overall ? EXIT_SUCCESS : EXIT_FAILURE;
}
