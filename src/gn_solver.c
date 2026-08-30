#include "nls_internal.h"

#include <math.h>

static double vec_dot(const double *a, const double *b, size_t n)
{
    double result = 0.0;
    size_t i;

    for (i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

static double vec_norm2(const double *values, size_t n)
{
    double scale = 0.0;
    double sum_squares = 1.0;
    size_t i;

    for (i = 0; i < n; ++i) {
        double magnitude = fabs(values[i]);
        if (magnitude == 0.0) {
            continue;
        }
        if (scale < magnitude) {
            double ratio = scale / magnitude;
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = magnitude;
        } else {
            double ratio = magnitude / scale;
            sum_squares += ratio * ratio;
        }
    }
    return scale == 0.0 ? 0.0 : scale * sqrt(sum_squares);
}

static int finite_array(const double *values, size_t n)
{
    size_t i;

    for (i = 0; i < n; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static void form_jtj(const double *jac, size_t m, size_t n, double *jtj)
{
    size_t i;
    size_t j;
    size_t k;

    for (j = 0; j < n; ++j) {
        for (k = 0; k < n; ++k) {
            double sum = 0.0;
            for (i = 0; i < m; ++i) {
                sum += jac[i * n + j] * jac[i * n + k];
            }
            jtj[j * n + k] = sum;
        }
    }
}

static void form_jtf(const double *jac, const double *fvec, size_t m, size_t n, double *jtf)
{
    size_t i;
    size_t j;

    for (j = 0; j < n; ++j) {
        double sum = 0.0;
        for (i = 0; i < m; ++i) {
            sum += jac[i * n + j] * fvec[i];
        }
        jtf[j] = sum;
    }
}

static int cholesky_solve(double *a, const double *rhs, size_t n, double *x)
{
    size_t i;
    size_t j;
    size_t k;

    for (i = 0; i < n; ++i) {
        for (j = 0; j <= i; ++j) {
            double sum = a[i * n + j];
            for (k = 0; k < j; ++k) {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if (i == j) {
                if (!(sum > 0.0) || !isfinite(sum)) {
                    return NLS_ERR_LINEAR;
                }
                a[i * n + j] = sqrt(sum);
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }

    for (i = 0; i < n; ++i) {
        double sum = -rhs[i];
        for (j = 0; j < i; ++j) {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }

    for (i = n; i-- > 0;) {
        double sum = x[i];
        for (j = i + 1; j < n; ++j) {
            sum -= a[j * n + i] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }

    return 0;
}

static int evaluate_problem(nls_solver *solver, const void *data, const double *params, double *fvec, double *jac)
{
    const nls_problem *problem = &solver->problem;

    ++solver->last_nfev;
    if (problem->f(data, problem->m, problem->n, params, fvec) != 0) {
        return NLS_ERR_CALLBACK;
    }
    if (!finite_array(fvec, problem->m)) {
        return NLS_ERR_NUMERIC;
    }
    ++solver->last_njev;
    if (problem->df(data, problem->m, problem->n, params, jac) != 0) {
        return NLS_ERR_CALLBACK;
    }
    if (!finite_array(jac, problem->m * problem->n)) {
        return NLS_ERR_NUMERIC;
    }
    return 0;
}

int nls_gn_solve(nls_solver *solver, const void *data, double *params)
{
    const nls_problem *problem = &solver->problem;
    double cost;
    size_t iteration;
    size_t j;
    int status;

    status = evaluate_problem(
        solver, data, params, solver->fvec, solver->jac_row);
    if (status != 0) {
        return status;
    }
    cost = 0.5 * vec_dot(solver->fvec, solver->fvec, problem->m);
    if (!isfinite(cost)) {
        return NLS_ERR_NUMERIC;
    }

    for (iteration = 0; iteration < problem->maxiter; ++iteration) {
        double new_cost;
        double step_norm;
        double x_norm;

        form_jtf(
            solver->jac_row,
            solver->fvec,
            problem->m,
            problem->n,
            solver->jtf);
        if (!finite_array(solver->jtf, problem->n)) {
            return NLS_ERR_NUMERIC;
        }
        {
            double gradient_norm = vec_norm2(solver->jtf, problem->n);
            if (!isfinite(gradient_norm)) {
                return NLS_ERR_NUMERIC;
            }
            if (gradient_norm <= problem->gtol) {
                return NLS_SUCCESS_GTOL;
            }
        }

        form_jtj(
            solver->jac_row,
            problem->m,
            problem->n,
            solver->jtj);
        if (!finite_array(solver->jtj, problem->n * problem->n)) {
            return NLS_ERR_NUMERIC;
        }
        status = cholesky_solve(
            solver->jtj, solver->jtf, problem->n, solver->step);
        if (status != 0) {
            return status;
        }
        if (!finite_array(solver->step, problem->n)) {
            return NLS_ERR_NUMERIC;
        }

        step_norm = vec_norm2(solver->step, problem->n);
        if (!isfinite(step_norm)) {
            return NLS_ERR_NUMERIC;
        }
        for (j = 0; j < problem->n; ++j) {
            params[j] += solver->step[j];
        }
        if (!finite_array(params, problem->n)) {
            return NLS_ERR_NUMERIC;
        }

        status = evaluate_problem(
            solver, data, params, solver->fvec, solver->jac_row);
        if (status != 0) {
            return status;
        }
        ++solver->last_iterations;
        new_cost = 0.5 * vec_dot(solver->fvec, solver->fvec, problem->m);
        if (!isfinite(new_cost)) {
            return NLS_ERR_NUMERIC;
        }

        x_norm = vec_norm2(params, problem->n);
        if (!isfinite(x_norm)) {
            return NLS_ERR_NUMERIC;
        }
        if (step_norm <= problem->xtol * (x_norm + problem->xtol)) {
            return NLS_SUCCESS_XTOL;
        }
        if (new_cost <= cost &&
            cost - new_cost <= problem->ftol * fmax(1.0, cost)) {
            return NLS_SUCCESS_FTOL;
        }
        cost = new_cost;
    }

    return NLS_MAX_ITER;
}
