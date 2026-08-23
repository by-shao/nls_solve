#include "nls_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int linear_f(
    const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    (void)data;
    (void)m;
    (void)n;
    fvec[0] = x[0] - 2.0;
    fvec[1] = x[1] - 3.0;
    return 0;
}

static int linear_j(
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

static int nonlinear_f(
    const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    (void)data;
    (void)m;
    (void)n;
    fvec[0] = x[0] * x[0] - 4.0;
    fvec[1] = x[1] - 3.0;
    return 0;
}

static int nonlinear_j(
    const void *data, size_t m, size_t n, const double *x, double *jac)
{
    (void)data;
    (void)m;
    (void)n;
    jac[0] = 2.0 * x[0];
    jac[1] = 0.0;
    jac[2] = 0.0;
    jac[3] = 1.0;
    return 0;
}

static int coupled_f(
    const void *data, size_t m, size_t n, const double *x, double *fvec)
{
    (void)data;
    (void)m;
    (void)n;
    fvec[0] = x[0] + 2.0 * x[1] - 8.0;
    fvec[1] = 3.0 * x[0] - x[1] - 3.0;
    fvec[2] = x[0] * x[1] - 6.0;
    return 0;
}

static int coupled_j(
    const void *data, size_t m, size_t n, const double *x, double *jac)
{
    (void)data;
    (void)m;
    (void)n;
    jac[0] = 1.0;
    jac[1] = 2.0;
    jac[2] = 3.0;
    jac[3] = -1.0;
    jac[4] = x[1];
    jac[5] = x[0];
    return 0;
}

static int near_solution(const double *x, double x0, double x1, double tol)
{
    return fabs(x[0] - x0) < tol && fabs(x[1] - x1) < tol;
}

static int solve_case(const char *name, eval_fvec f, eval_fjac j, size_t m, NlsAlgorithm algorithm, const double initial[2])
{
    nls_problem problem;
    nls_solver *solver;
    double x[2] = {initial[0], initial[1]};
    int status;

    if (nls_problem_init(&problem, f, j, m, 2) != 0) {
        return 1;
    }
    solver = nls_solver_alloc(&problem, algorithm, LLS_ALGO_CHOLESKY);
    if (solver == NULL) {
        return 1;
    }
    status = nls_solve(solver, NULL, x);
    nls_solver_free(solver);

    printf("%s %s: status=%d x=[%.12g, %.12g]\n",
           name,
           algorithm == NLS_ALGO_LM ? "LM" : "GN",
           status,
           x[0],
           x[1]);
    return status <= 0 || !near_solution(x, 2.0, 3.0, 1e-8);
}

static int test_finite_difference(void)
{
    const double x[2] = {1.4, -0.7};
    const double step = 1e-6;
    double analytic[6];
    double numeric[6];
    double xp[2] = {x[0], x[1]};
    double xm[2] = {x[0], x[1]};
    double fp[3];
    double fm[3];
    double max_error = 0.0;
    size_t i;
    size_t j;

    if (coupled_j(NULL, 3, 2, x, analytic) != 0) {
        return 1;
    }
    for (j = 0; j < 2; ++j) {
        xp[0] = xm[0] = x[0];
        xp[1] = xm[1] = x[1];
        xp[j] += step;
        xm[j] -= step;
        coupled_f(NULL, 3, 2, xp, fp);
        coupled_f(NULL, 3, 2, xm, fm);
        for (i = 0; i < 3; ++i) {
            double error;
            numeric[i * 2 + j] = (fp[i] - fm[i]) / (2.0 * step);
            error = fabs(analytic[i * 2 + j] - numeric[i * 2 + j]);
            if (error > max_error) {
                max_error = error;
            }
        }
    }

    printf("NLS-3 row-major finite difference: max_error=%.3g\n", max_error);
    return max_error >= 1e-6;
}

int main(void)
{
    const double zero[2] = {0.0, 0.0};
    const double nonlinear_initial[2] = {1.0, 0.0};
    int failed = 0;

    failed |= solve_case("NLS-1", linear_f, linear_j, 2, NLS_ALGO_LM, zero);
    failed |= solve_case("NLS-1", linear_f, linear_j, 2, NLS_ALGO_GN, zero);
    failed |= solve_case(
        "NLS-2", nonlinear_f, nonlinear_j, 2, NLS_ALGO_LM, nonlinear_initial);
    failed |= solve_case(
        "NLS-2", nonlinear_f, nonlinear_j, 2, NLS_ALGO_GN, nonlinear_initial);
    failed |= test_finite_difference();
    failed |= solve_case("NLS-3", coupled_f, coupled_j, 3, NLS_ALGO_LM, zero);

    if (failed) {
        fprintf(stderr, "test_nls failed\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
