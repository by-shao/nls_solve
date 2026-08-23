#include "nls_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cminpack.h>

typedef struct {
    const nls_problem *problem;
    const void *user_data;
    double *row_jac;
    int failure_status;
} LmAdapterData;

static int size_product_ok(size_t a, size_t b)
{
    return a == 0 || b <= SIZE_MAX / a;
}

static void *checked_calloc(size_t count, size_t size)
{
    if (!size_product_ok(count, size)) {
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

static int lmder_adapter(
    void *p,
    int m,
    int n,
    const double *x,
    double *fvec,
    double *fjac,
    int ldfjac,
    int iflag)
{
    LmAdapterData *adapter = (LmAdapterData *)p;
    size_t i;
    size_t j;
    int callback_status;

    if (adapter == NULL || adapter->problem == NULL || m <= 0 || n <= 0) {
        return -1;
    }

    if (iflag == 1) {
        callback_status = adapter->problem->f(
            adapter->user_data, (size_t)m, (size_t)n, x, fvec);
        if (callback_status != 0) {
            adapter->failure_status = NLS_ERR_CALLBACK;
            return -1;
        }
        if (!finite_array(fvec, (size_t)m)) {
            adapter->failure_status = NLS_ERR_NUMERIC;
            return -1;
        }
        return 0;
    }

    if (iflag == 2) {
        callback_status = adapter->problem->df(
            adapter->user_data, (size_t)m, (size_t)n, x, adapter->row_jac);
        if (callback_status != 0) {
            adapter->failure_status = NLS_ERR_CALLBACK;
            return -1;
        }
        if (!finite_array(adapter->row_jac, (size_t)m * (size_t)n)) {
            adapter->failure_status = NLS_ERR_NUMERIC;
            return -1;
        }
        for (i = 0; i < (size_t)m; ++i) {
            for (j = 0; j < (size_t)n; ++j) {
                fjac[i + j * (size_t)ldfjac] =
                    adapter->row_jac[i * (size_t)n + j];
            }
        }
        return 0;
    }

    return 0;
}

static int map_lmder_info(int info, int adapter_failure)
{
    if (info < 0) {
        return adapter_failure != 0 ? adapter_failure : NLS_ERR_CALLBACK;
    }

    switch (info) {
    case 0:
        return NLS_ERR_INVALID;
    case 1:
    case 3:
        return NLS_SUCCESS_FTOL;
    case 2:
        return NLS_SUCCESS_XTOL;
    case 4:
    case 8:
        return NLS_SUCCESS_GTOL;
    case 5:
        return NLS_MAX_ITER;
    case 6:
    case 7:
        return NLS_ERR_NUMERIC;
    default:
        return NLS_ERR_NUMERIC;
    }
}

static int validate_problem_for_solve(const nls_problem *problem)
{
    size_t i;

    if (problem == NULL || problem->f == NULL || problem->df == NULL ||
        problem->m == 0 || problem->n == 0 || problem->maxiter == 0 ||
        problem->m > (size_t)INT_MAX || problem->n > (size_t)INT_MAX ||
        problem->maxiter > (size_t)INT_MAX ||
        !isfinite(problem->ftol) || problem->ftol < 0.0 ||
        !isfinite(problem->xtol) || problem->xtol < 0.0 ||
        !isfinite(problem->gtol) || problem->gtol < 0.0 ||
        !isfinite(problem->factor) || problem->factor <= 0.0) {
        return 0;
    }

    if (problem->diag != NULL) {
        for (i = 0; i < problem->n; ++i) {
            if (!isfinite(problem->diag[i]) || problem->diag[i] <= 0.0) {
                return 0;
            }
        }
    }

    return 1;
}

int nls_problem_init(
    nls_problem *prob,
    eval_fvec f,
    eval_fjac df,
    size_t m,
    size_t n)
{
    if (prob == NULL || f == NULL || df == NULL || m == 0 || n == 0) {
        return NLS_ERR_INVALID;
    }

    prob->f = f;
    prob->df = df;
    prob->m = m;
    prob->n = n;
    prob->ftol = sqrt(DBL_EPSILON);
    prob->xtol = sqrt(DBL_EPSILON);
    prob->gtol = 0.0;
    prob->factor = 100.0;
    prob->epsfcn = 0.0;
    prob->diag = NULL;
    prob->maxiter = 20;

    return 0;
}

nls_solver *nls_solver_alloc(
    nls_problem *prob,
    NlsAlgorithm nls,
    LlsAlgorithm lls)
{
    nls_solver *solver;
    size_t mn;

    if (!validate_problem_for_solve(prob) || nls < 0 || nls >= NLS_ALGO_MAX ||
        lls < 0 || lls >= LLS_ALGO_MAX ||
        (nls == NLS_ALGO_LM && prob->m < prob->n) ||
        !size_product_ok(prob->m, prob->n) ||
        !size_product_ok(prob->n, prob->n)) {
        return NULL;
    }

    solver = (nls_solver *)calloc(1, sizeof(*solver));
    if (solver == NULL) {
        return NULL;
    }

    solver->problem = *prob;
    solver->nls_algorithm = nls;
    solver->lls_algorithm = lls;
    mn = prob->m * prob->n;

    solver->fvec = (double *)checked_calloc(prob->m, sizeof(double));
    solver->jac_row = (double *)checked_calloc(mn, sizeof(double));
    if (prob->diag != NULL) {
        solver->diag_config =
            (double *)checked_calloc(prob->n, sizeof(double));
        if (solver->diag_config != NULL) {
            memcpy(
                solver->diag_config, prob->diag, prob->n * sizeof(double));
            solver->problem.diag = solver->diag_config;
        }
    }

    if (nls == NLS_ALGO_LM) {
        solver->fjac = (double *)checked_calloc(mn, sizeof(double));
        solver->diag = (double *)checked_calloc(prob->n, sizeof(double));
        solver->qtf = (double *)checked_calloc(prob->n, sizeof(double));
        solver->wa1 = (double *)checked_calloc(prob->n, sizeof(double));
        solver->wa2 = (double *)checked_calloc(prob->n, sizeof(double));
        solver->wa3 = (double *)checked_calloc(prob->n, sizeof(double));
        solver->wa4 = (double *)checked_calloc(prob->m, sizeof(double));
        solver->ipvt = (int *)checked_calloc(prob->n, sizeof(int));
    } else {
        solver->jtj = (double *)checked_calloc(prob->n * prob->n, sizeof(double));
        solver->jtf = (double *)checked_calloc(prob->n, sizeof(double));
        solver->step = (double *)checked_calloc(prob->n, sizeof(double));
    }

    if (solver->fvec == NULL || solver->jac_row == NULL ||
        (prob->diag != NULL && solver->diag_config == NULL) ||
        (nls == NLS_ALGO_LM &&
         (solver->fjac == NULL || solver->diag == NULL || solver->qtf == NULL ||
          solver->wa1 == NULL || solver->wa2 == NULL || solver->wa3 == NULL ||
          solver->wa4 == NULL || solver->ipvt == NULL)) ||
        (nls == NLS_ALGO_GN &&
         (solver->jtj == NULL || solver->jtf == NULL || solver->step == NULL))) {
        nls_solver_free(solver);
        return NULL;
    }

    return solver;
}

static int solve_lm(nls_solver *solver, const void *data, double *params)
{
    const nls_problem *problem = &solver->problem;
    LmAdapterData adapter;
    int info;
    int mode;
    int nfev = 0;
    int njev = 0;

    if (problem->m < problem->n) {
        return NLS_ERR_INVALID;
    }

    memset(solver->fvec, 0, problem->m * sizeof(double));
    memset(solver->fjac, 0, problem->m * problem->n * sizeof(double));
    memset(solver->jac_row, 0, problem->m * problem->n * sizeof(double));
    memset(solver->qtf, 0, problem->n * sizeof(double));
    memset(solver->wa1, 0, problem->n * sizeof(double));
    memset(solver->wa2, 0, problem->n * sizeof(double));
    memset(solver->wa3, 0, problem->n * sizeof(double));
    memset(solver->wa4, 0, problem->m * sizeof(double));
    memset(solver->ipvt, 0, problem->n * sizeof(int));

    if (problem->diag == NULL) {
        memset(solver->diag, 0, problem->n * sizeof(double));
        mode = 1;
    } else {
        memcpy(solver->diag, problem->diag, problem->n * sizeof(double));
        mode = 2;
    }

    adapter.problem = problem;
    adapter.user_data = data;
    adapter.row_jac = solver->jac_row;
    adapter.failure_status = 0;

    info = lmder(
        lmder_adapter,
        &adapter,
        (int)problem->m,
        (int)problem->n,
        params,
        solver->fvec,
        solver->fjac,
        (int)problem->m,
        problem->ftol,
        problem->xtol,
        problem->gtol,
        (int)problem->maxiter,
        solver->diag,
        mode,
        problem->factor,
        0,
        &nfev,
        &njev,
        solver->ipvt,
        solver->qtf,
        solver->wa1,
        solver->wa2,
        solver->wa3,
        solver->wa4);

    return map_lmder_info(info, adapter.failure_status);
}

int nls_solve(
    nls_solver *solver,
    const void *data,
    double *params)
{
    if (solver == NULL || params == NULL ||
        !validate_problem_for_solve(&solver->problem) ||
        !finite_array(params, solver->problem.n)) {
        return NLS_ERR_INVALID;
    }

    if (solver->nls_algorithm == NLS_ALGO_LM) {
        return solve_lm(solver, data, params);
    }
    if (solver->nls_algorithm == NLS_ALGO_GN) {
        if (solver->lls_algorithm != LLS_ALGO_CHOLESKY) {
            return NLS_ERR_INVALID;
        }
        return nls_gn_solve(solver, data, params);
    }
    return NLS_ERR_INVALID;
}

void nls_solver_free(
    nls_solver *solver)
{
    if (solver == NULL) {
        return;
    }

    free(solver->fvec);
    free(solver->fjac);
    free(solver->jac_row);
    free(solver->diag);
    free(solver->diag_config);
    free(solver->qtf);
    free(solver->wa1);
    free(solver->wa2);
    free(solver->wa3);
    free(solver->wa4);
    free(solver->ipvt);
    free(solver->jtj);
    free(solver->jtf);
    free(solver->step);
    free(solver);
}
