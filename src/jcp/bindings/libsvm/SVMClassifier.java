// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.libsvm;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import jcp.ml.IClassifier;

public class SVMClassifier
    implements IClassifier,
               java.io.Serializable
{
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected svm_parameter _parameter;
    protected svm_model _model;

    public SVMClassifier()
    {
        this(null);
    }

    public SVMClassifier(svm_parameter parameter)
    {
        _parameter = parameter;

        if (_parameter == null) {
            // Default libsvm parameter.
            _parameter = new svm_parameter();
            _parameter.svm_type = svm_parameter.C_SVC;
            _parameter.kernel_type = svm_parameter.RBF;
            _parameter.degree = 3;
            _parameter.gamma = 1.0/2; // FIXME: Should be 1/#classes
            _parameter.coef0 = 0;
            _parameter.nu = 0.5;
            _parameter.cache_size = 100;
            _parameter.C = 1;
            _parameter.eps = 1e-3;
            _parameter.p = 0.1;
            _parameter.shrinking = 1;
            _parameter.probability = 1; // Must be set for the NC-function.
            _parameter.nr_weight = 0;
            _parameter.weight_label = new int[0];
            _parameter.weight = new double[0];
        }
    }

    public void fit(DoubleMatrix2D x, double[] y)
    {
        SparseDoubleMatrix2D tmp_x;
        if (x instanceof jcp.bindings.libsvm.SparseDoubleMatrix2D) {
            tmp_x = (SparseDoubleMatrix2D)x;
        } else {
            tmp_x = new SparseDoubleMatrix2D(x.rows(), x.columns());
            tmp_x.assign(x);
        }

        _model = svm.svm_train(_parameter, tmp_x, y);
    }

    public double predict(DoubleMatrix1D instance)
    {
        SparseDoubleMatrix1D tmp_instance;
        if (instance instanceof jcp.bindings.libsvm.SparseDoubleMatrix1D) {
            tmp_instance = (SparseDoubleMatrix1D)instance;
        } else {
            tmp_instance = new SparseDoubleMatrix1D(instance.size());
            tmp_instance.assign(instance);
        }

        return svm.svm_predict(_model, tmp_instance);
    }

    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates)
    {
        SparseDoubleMatrix1D tmp_instance;
        if (instance instanceof jcp.bindings.libsvm.SparseDoubleMatrix1D) {
            tmp_instance = (SparseDoubleMatrix1D)instance;
        } else {
            tmp_instance = new SparseDoubleMatrix1D(instance.size());
            tmp_instance.assign(instance);
        }

        return svm.svm_predict_probability(_model,
                                           tmp_instance,
                                           probabilityEstimates);
    }

    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }
}
