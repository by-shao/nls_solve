#include "auglag_constrain.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DG_PARAMETER_COUNT 7
#define INPUT_MAGIC "NLS_DG_INPUT_V1"

typedef struct {
    size_t m;
    const double *h;
    const double *v;
    const double *observed;
} DoubleGaussianData;

typedef struct {
    size_t parameter_index;
    double lower;
} LowerBound;

typedef struct {
    char name[32];
    size_t grid_nx;
    size_t grid_ny;
    size_t full_count;
    size_t mask_count;
    size_t m;
    size_t *original_index;
    double *h;
    double *v;
    double *observed;
    double truth[DG_PARAMETER_COUNT];
    double initial[DG_PARAMETER_COUNT];
    double lower[DG_PARAMETER_COUNT];
} DoubleGaussianCase;

typedef struct {
    int status;
    double params[DG_PARAMETER_COUNT];
    double sse;
    double rmse;
    double violation;
    double max_h;
    double max_v;
    int finite;
    double rho;
    size_t outer_iterations;
    size_t inner_iterations;
    size_t inner_nfev;
    size_t inner_njev;
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

static void free_case(DoubleGaussianCase *test_case)
{
    if (test_case == NULL) {
        return;
    }
    free(test_case->original_index);
    free(test_case->h);
    free(test_case->v);
    free(test_case->observed);
    memset(test_case, 0, sizeof(*test_case));
}

static int expect_token(FILE *input, const char *expected)
{
    char token[64];

    return fscanf(input, "%63s", token) == 1 &&
        strcmp(token, expected) == 0;
}

static int read_vector(FILE *input, double *values, size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        if (fscanf(input, "%lf", &values[i]) != 1) {
            return 0;
        }
    }
    return 1;
}

static int read_case(const char *path, DoubleGaussianCase *test_case)
{
    char magic[64];
    FILE *input = fopen(path, "r");
    size_t i;
    int ok = 0;

    if (input == NULL || test_case == NULL) {
        if (input != NULL) {
            fclose(input);
        }
        return 0;
    }
    memset(test_case, 0, sizeof(*test_case));
    if (fscanf(input, "%63s", magic) != 1 || strcmp(magic, INPUT_MAGIC) != 0 ||
        !expect_token(input, "name") ||
        fscanf(input, "%31s", test_case->name) != 1 ||
        !expect_token(input, "grid_nx") ||
        fscanf(input, "%zu", &test_case->grid_nx) != 1 ||
        !expect_token(input, "grid_ny") ||
        fscanf(input, "%zu", &test_case->grid_ny) != 1 ||
        !expect_token(input, "full_count") ||
        fscanf(input, "%zu", &test_case->full_count) != 1 ||
        !expect_token(input, "mask_count") ||
        fscanf(input, "%zu", &test_case->mask_count) != 1 ||
        !expect_token(input, "m") || fscanf(input, "%zu", &test_case->m) != 1 ||
        !expect_token(input, "truth") ||
        !read_vector(input, test_case->truth, DG_PARAMETER_COUNT) ||
        !expect_token(input, "initial") ||
        !read_vector(input, test_case->initial, DG_PARAMETER_COUNT) ||
        !expect_token(input, "lower") ||
        !read_vector(input, test_case->lower, DG_PARAMETER_COUNT) ||
        !expect_token(input, "data") || test_case->m == 0 ||
        test_case->mask_count + test_case->m != test_case->full_count) {
        goto done;
    }

    test_case->original_index =
        (size_t *)calloc(test_case->m, sizeof(*test_case->original_index));
    test_case->h = (double *)calloc(test_case->m, sizeof(*test_case->h));
    test_case->v = (double *)calloc(test_case->m, sizeof(*test_case->v));
    test_case->observed =
        (double *)calloc(test_case->m, sizeof(*test_case->observed));
    if (test_case->original_index == NULL || test_case->h == NULL ||
        test_case->v == NULL || test_case->observed == NULL) {
        goto done;
    }
    for (i = 0; i < test_case->m; ++i) {
        if (fscanf(
                input,
                "%zu %lf %lf %lf",
                &test_case->original_index[i],
                &test_case->h[i],
                &test_case->v[i],
                &test_case->observed[i]) != 4 ||
            test_case->original_index[i] >= test_case->full_count ||
            !isfinite(test_case->h[i]) || !isfinite(test_case->v[i]) ||
            !isfinite(test_case->observed[i])) {
            goto done;
        }
    }
    ok = 1;

done:
    fclose(input);
    if (!ok) {
        free_case(test_case);
    }
    return ok;
}

static uint64_t hash_u64(uint64_t hash, uint64_t value)
{
    hash ^= value;
    return hash * UINT64_C(1099511628211);
}

static uint64_t hash_double(uint64_t hash, double value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return hash_u64(hash, bits);
}

static uint64_t input_hash(const DoubleGaussianCase *test_case)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;

    hash = hash_u64(hash, 1);
    hash = hash_u64(hash, (uint64_t)test_case->grid_nx);
    hash = hash_u64(hash, (uint64_t)test_case->grid_ny);
    hash = hash_u64(hash, (uint64_t)test_case->full_count);
    hash = hash_u64(hash, (uint64_t)test_case->mask_count);
    hash = hash_u64(hash, (uint64_t)test_case->m);
    for (i = 0; i < DG_PARAMETER_COUNT; ++i) {
        hash = hash_double(hash, test_case->truth[i]);
    }
    for (i = 0; i < DG_PARAMETER_COUNT; ++i) {
        hash = hash_double(hash, test_case->initial[i]);
    }
    for (i = 0; i < DG_PARAMETER_COUNT; ++i) {
        hash = hash_double(hash, test_case->lower[i]);
    }
    for (i = 0; i < test_case->m; ++i) {
        hash = hash_u64(hash, (uint64_t)test_case->original_index[i]);
        hash = hash_double(hash, test_case->h[i]);
        hash = hash_double(hash, test_case->v[i]);
        hash = hash_double(hash, test_case->observed[i]);
    }
    return hash;
}

static int jacobian_check(const DoubleGaussianCase *test_case, double *max_error, double *relative_error)
{
    DoubleGaussianData data = {
        test_case->m, test_case->h, test_case->v, test_case->observed
    };
    const size_t mn = test_case->m * DG_PARAMETER_COUNT;
    double *analytic = (double *)calloc(mn, sizeof(*analytic));
    double *plus = (double *)calloc(test_case->m, sizeof(*plus));
    double *minus = (double *)calloc(test_case->m, sizeof(*minus));
    double xp[DG_PARAMETER_COUNT];
    double xm[DG_PARAMETER_COUNT];
    size_t i;
    size_t j;
    int ok = 0;

    if (analytic == NULL || plus == NULL || minus == NULL) {
        goto done;
    }
    memcpy(xp, test_case->initial, sizeof(xp));
    memcpy(xm, test_case->initial, sizeof(xm));
    if (double_gaussian_jacobian(
            &data,
            test_case->m,
            DG_PARAMETER_COUNT,
            test_case->initial,
            analytic) != 0) {
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
                &data, test_case->m, DG_PARAMETER_COUNT, xp, plus) != 0 ||
            double_gaussian_residual(
                &data, test_case->m, DG_PARAMETER_COUNT, xm, minus) != 0) {
            goto done;
        }
        for (i = 0; i < test_case->m; ++i) {
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

static int evaluate_result(const DoubleGaussianCase *test_case, DoubleGaussianResult *result)
{
    DoubleGaussianData data = {
        test_case->m, test_case->h, test_case->v, test_case->observed
    };
    double *residual = (double *)calloc(test_case->m, sizeof(*residual));
    size_t i;

    if (residual == NULL ||
        double_gaussian_residual(
            &data,
            test_case->m,
            DG_PARAMETER_COUNT,
            result->params,
            residual) != 0) {
        free(residual);
        return 0;
    }
    result->sse = 0.0;
    for (i = 0; i < test_case->m; ++i) {
        result->sse += residual[i] * residual[i];
    }
    result->rmse = sqrt(result->sse / (double)test_case->m);
    result->violation = constraint_violation(test_case, result->params);
    result->max_h = -(result->params[5] + result->params[6]) /
        (2.0 * result->params[3]);
    result->max_v = -(result->params[5] - result->params[6]) /
        (2.0 * result->params[4]);
    result->finite = finite_array(result->params, DG_PARAMETER_COUNT) &&
        isfinite(result->sse) && isfinite(result->rmse) &&
        isfinite(result->violation) && isfinite(result->max_h) &&
        isfinite(result->max_v) && isfinite(result->rho);
    free(residual);
    return result->finite;
}

static void solve_case(const DoubleGaussianCase *test_case, NlsAlgorithm algorithm, DoubleGaussianResult *result)
{
    DoubleGaussianData data = {
        test_case->m, test_case->h, test_case->v, test_case->observed
    };
    AugLagProblem problem = {
        double_gaussian_residual,
        double_gaussian_jacobian,
        &data,
        test_case->m,
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
    options.constraint_tol = 1.0e-8;

    result->status = auglag_init(
        &context,
        &problem,
        &constraints,
        &options,
        NULL,
        algorithm,
        algorithm == NLS_ALGO_LM ? LLS_ALGO_QR : LLS_ALGO_CHOLESKY);
    if (result->status != 0) {
        result->finite = finite_array(result->params, DG_PARAMETER_COUNT);
        return;
    }
    result->status = auglag_solve(&context, result->params);
    result->rho = context.rho;
    result->outer_iterations = context.outer_iterations;
    result->inner_iterations = context.inner_iterations;
    result->inner_nfev = context.inner_function_evaluations;
    result->inner_njev = context.inner_jacobian_evaluations;
    (void)evaluate_result(test_case, result);
    auglag_destroy(&context);
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

static void print_result(const DoubleGaussianCase *test_case, const char *algorithm, uint64_t hash, double jac_error, double jac_relative_error, const DoubleGaussianResult *result)
{
    printf("C_RESULT_BEGIN\n");
    printf("case=%s\n", test_case->name);
    printf("algorithm=%s\n", algorithm);
    printf("input_hash=%016" PRIx64 "\n", hash);
    printf("status=%d\n", result->status);
    printf("final=");
    print_vector(result->params, DG_PARAMETER_COUNT);
    printf("\n");
    printf("sse=%.17g\n", result->sse);
    printf("rmse=%.17g\n", result->rmse);
    printf("constraint_violation=%.17g\n", result->violation);
    printf("finite_check=%s\n", result->finite ? "PASS" : "FAIL");
    printf("max_point=[%.17g,%.17g]\n", result->max_h, result->max_v);
    printf("rho_final=%.17g\n", result->rho);
    printf("outer_iterations=%zu\n", result->outer_iterations);
    printf("inner_iterations=%zu\n", result->inner_iterations);
    printf("inner_nfev=%zu\n", result->inner_nfev);
    printf("inner_njev=%zu\n", result->inner_njev);
    printf("jac_error=%.17g\n", jac_error);
    printf("jac_relative_error=%.17g\n", jac_relative_error);
    printf("C_RESULT_END\n");
}

int main(int argc, char **argv)
{
    DoubleGaussianCase test_case;
    DoubleGaussianResult lm_result;
    DoubleGaussianResult gn_result;
    uint64_t hash;
    double jac_error = NAN;
    double jac_relative_error = NAN;

    if (argc != 2) {
        fprintf(stderr, "usage: %s INPUT_FILE\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (!read_case(argv[1], &test_case)) {
        fprintf(stderr, "invalid Double Gaussian input: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    hash = input_hash(&test_case);
    if (!jacobian_check(&test_case, &jac_error, &jac_relative_error)) {
        fprintf(stderr, "Double Gaussian Jacobian check could not run\n");
    }
    solve_case(&test_case, NLS_ALGO_LM, &lm_result);
    solve_case(&test_case, NLS_ALGO_GN, &gn_result);
    print_result(
        &test_case,
        "LM",
        hash,
        jac_error,
        jac_relative_error,
        &lm_result);
    print_result(
        &test_case,
        "GN",
        hash,
        jac_error,
        jac_relative_error,
        &gn_result);
    free_case(&test_case);
    return EXIT_SUCCESS;
}
