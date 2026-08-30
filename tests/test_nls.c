#include "nls_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    CORE_M = 3,
    CORE_N = 2
};

typedef struct {
    double rhs[CORE_M];
} CoupledProblemData;

static int coupled_residual(const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    const CoupledProblemData *problem = (const CoupledProblemData *)data;

    if (problem == NULL || x == NULL || fvec == NULL ||
        m != CORE_M || n != CORE_N) {
        return 1;
    }

    fvec[0] = x[0] + 2.0 * x[1] - problem->rhs[0];
    fvec[1] = 3.0 * x[0] - x[1] - problem->rhs[1];
    fvec[2] = x[0] * x[1] - problem->rhs[2];
    return 0;
}

static int coupled_jacobian(const void *data, size_t m, size_t n, const double *x, double *jac)
{
    if (data == NULL || x == NULL || jac == NULL ||
        m != CORE_M || n != CORE_N) {
        return 1;
    }

    /* Row-major: jac[residual_index * n + parameter_index]. */
    jac[0 * CORE_N + 0] = 1.0;
    jac[0 * CORE_N + 1] = 2.0;
    jac[1 * CORE_N + 0] = 3.0;
    jac[1 * CORE_N + 1] = -1.0;
    jac[2 * CORE_N + 0] = x[1];
    jac[2 * CORE_N + 1] = x[0];
    return 0;
}

static int check_row_major_jacobian(const CoupledProblemData *data)
{
    const double x[CORE_N] = {1.4, -0.7};
    double analytic[CORE_M * CORE_N];
    double numeric[CORE_M * CORE_N];
    double max_error = 0.0;
    size_t i;
    size_t j;

    if (coupled_jacobian(data, CORE_M, CORE_N, x, analytic) != 0) {
        return 1;
    }

    for (j = 0; j < CORE_N; ++j) {
        double xp[CORE_N] = {x[0], x[1]};
        double xm[CORE_N] = {x[0], x[1]};
        double fp[CORE_M];
        double fm[CORE_M];
        const double step = 1.0e-6 * fmax(1.0, fabs(x[j]));

        xp[j] += step;
        xm[j] -= step;
        if (coupled_residual(data, CORE_M, CORE_N, xp, fp) != 0 ||
            coupled_residual(data, CORE_M, CORE_N, xm, fm) != 0) {
            return 1;
        }
        for (i = 0; i < CORE_M; ++i) {
            const size_t index = i * CORE_N + j;
            double error;

            numeric[index] = (fp[i] - fm[i]) / (2.0 * step);
            error = fabs(analytic[index] - numeric[index]);
            if (error > max_error) {
                max_error = error;
            }
        }
    }

    printf("CORE1 row-major central-FD: max_error=%.3g\n", max_error);
    return !isfinite(max_error) || max_error >= 1.0e-8;
}

static int solve_core_case(nls_problem *problem, const CoupledProblemData *data, NlsAlgorithm algorithm)
{
    nls_solver *solver;
    double x[CORE_N] = {1.0, 1.0};
    double residual[CORE_M];
    double max_residual = 0.0;
    size_t i;
    int status;

    solver = nls_solver_alloc(problem, algorithm, LLS_ALGO_CHOLESKY);
    if (solver == NULL) {
        fprintf(stderr, "CORE1 %s allocation failed\n",
                algorithm == NLS_ALGO_LM ? "LM" : "GN");
        return 1;
    }

    status = nls_solve(solver, data, x);
    nls_solver_free(solver);

    if (coupled_residual(data, CORE_M, CORE_N, x, residual) != 0) {
        return 1;
    }
    for (i = 0; i < CORE_M; ++i) {
        max_residual = fmax(max_residual, fabs(residual[i]));
    }

    printf("CORE1 %s: status=%d x=[%.12g, %.12g] max_residual=%.3g\n",
           algorithm == NLS_ALGO_LM ? "LM" : "GN",
           status,
           x[0],
           x[1],
           max_residual);

    return status <= NLS_MAX_ITER || !isfinite(x[0]) || !isfinite(x[1]) ||
           fabs(x[0] - 2.0) >= 1.0e-8 ||
           fabs(x[1] - 3.0) >= 1.0e-8 || max_residual >= 1.0e-8;
}

static int check_gn_qr_rejected(nls_problem *problem)
{
    nls_solver *solver =
        nls_solver_alloc(problem, NLS_ALGO_GN, LLS_ALGO_QR);

    if (solver != NULL) {
        fprintf(stderr, "CORE1 GN+QR was not rejected during allocation\n");
        nls_solver_free(solver);
        return 1;
    }

    printf("CORE1 GN+QR: rejected during allocation\n");
    return 0;
}

int main(void)
{
    const CoupledProblemData data = {{8.0, 3.0, 6.0}};
    nls_problem problem;
    int failed = 0;

    if (nls_problem_init(
            &problem, coupled_residual, coupled_jacobian, CORE_M, CORE_N) != 0) {
        fprintf(stderr, "CORE1 nls_problem_init failed\n");
        return EXIT_FAILURE;
    }

    failed |= check_row_major_jacobian(&data);
    failed |= check_gn_qr_rejected(&problem);
    failed |= solve_core_case(&problem, &data, NLS_ALGO_LM);
    failed |= solve_core_case(&problem, &data, NLS_ALGO_GN);

    if (failed) {
        fprintf(stderr, "test_nls CORE1 failed\n");
        return EXIT_FAILURE;
    }

    printf("test_nls CORE1 passed\n");
    return EXIT_SUCCESS;
}
