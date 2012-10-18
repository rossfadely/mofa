#include <Python.h>
#include <numpy/arrayobject.h>

PyMODINIT_FUNC init_algorithms(void);
static PyObject *algorithms_kmeans(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"kmeans", algorithms_kmeans, METH_VARARGS, "Faster K-means."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_algorithms(void)
{
    PyObject *m = Py_InitModule("_algorithms", module_methods);
    if (m == NULL)
        return;
    import_array(); /* Load NumPy */
}

static PyObject *algorithms_kmeans(PyObject *self, PyObject *args)
{
    /* SHAPES:
        data  -> (P, D)
        means -> (K, D)
        rs    -> (P,)
     */

    /* parse the input tuple */
    PyObject *data_obj = NULL, *means_obj = NULL, *rs_obj = NULL;
    double tol;
    int maxiter;
    if (!PyArg_ParseTuple(args, "OOOdi", &data_obj, &means_obj, &rs_obj, &tol, &maxiter))
        return NULL;

    /* get numpy arrays */
    PyObject *data_array  = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *means_array = PyArray_FROM_OTF(means_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *rs_array    = PyArray_FROM_OTF(rs_obj, NPY_INTP, NPY_INOUT_ARRAY);
    if (data_array == NULL || means_array == NULL || rs_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(data_array);
        Py_XDECREF(means_array);
        Py_XDECREF(rs_array);
        return NULL;
    }

    double *data  = (double*)PyArray_DATA(data_array);
    double *means = (double*)PyArray_DATA(means_array);
    long   *rs    = (long*)PyArray_DATA(rs_array);

    int p, d, k;
    int P = (int)PyArray_DIM(data_array, 0);
    int D = (int)PyArray_DIM(data_array, 1);
    int K = (int)PyArray_DIM(means_array, 0);

    double *dists = (double*)malloc(K*sizeof(double));
    long   *N_rs  = (long*)malloc(K*sizeof(long));

    double L = 1.0;
    int iter;
    for (iter = 0; iter < maxiter; iter++) {
        double L_new = 0.0, dL;
        for (p = 0; p < P; p++) {
            double min_dist = -1.0;
            for (k = 0; k < K; k++) {
                dists[k] = 0.0;
                for (d = 0; d < D; d++) {
                    double diff = means[k*D+d] - data[p*D+d];
                    dists[k] += diff*diff;
                }
                if (min_dist < 0 || dists[k] < min_dist) {
                    min_dist = dists[k];
                    rs[p] = k;
                }
            }
            L_new += dists[rs[p]];
        }

        /* check for convergence */
        dL = fabs(L_new - L)/L;
        if (iter > 5 && dL < tol)
            break;
        else
            L = L_new;

        /* update means */
        for (k = 0; k < K; k++)
            N_rs[k] = 0;
        for (p = 0; p < P; p++) {
            N_rs[rs[p]] += 1;

            for (d = 0; d < D; d++) {
                means[rs[p]*D + d] += data[p*D + d];
            }
        }

        for (k = 0; k < K; k++) {
            for (d = 0; d < D; d++) {
                means[k*D + d] /= (double)N_rs[k];
            }
        }
    }

    /* clean up */
    Py_DECREF(data_array);
    Py_DECREF(means_array);
    Py_DECREF(rs_array);
    free(dists);
    free(N_rs);

    /* return None */
    PyObject *ret = Py_BuildValue("i", iter);
    return ret;
}
