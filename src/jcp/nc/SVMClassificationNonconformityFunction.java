// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.nc;

import java.util.Map;
import java.util.TreeMap;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import jcp.bindings.libsvm.*;

public class SVMClassificationNonconformityFunction
    implements IClassificationNonconformityFunction
{
    private static final boolean DEBUG = false;

    svm_parameter _parameter;
    svm_model _model;
    int _n_classes;
    double[] _classes;
    Map<Double, Integer> _class_index = new TreeMap<Double, Integer>();

    public SVMClassificationNonconformityFunction(double[] classes)
    {
        this(classes, null);
    }

    public SVMClassificationNonconformityFunction(double[] classes,
                                                  svm_parameter parameter)
    {
        _parameter = parameter;
        _n_classes = classes.length;
        _classes = classes;

        for (int i = 0; i < _n_classes; i++) {
            _class_index.put(_classes[i], i);
        }

        if (_parameter == null) {
            // Default libsvm parameter.
            _parameter = new svm_parameter();
            _parameter.svm_type = svm_parameter.C_SVC;
            _parameter.kernel_type = svm_parameter.RBF;
            _parameter.degree = 3;
            _parameter.gamma = 1.0/classes.length;
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

    public void fit(DoubleMatrix2D x,
                    double[] y)
    {
        if (x instanceof jcp.bindings.libsvm.SparseDoubleMatrix2D) {
            fastFit((jcp.bindings.libsvm.SparseDoubleMatrix2D)x, y);
        } else {
            slowFit(x, y);
        }
    }

    
    public void fit(DoubleMatrix2D xtr,
                    double[] ytr,
                    DoubleMatrix1D xtest,
                    double ytest)
    {
        throw new UnsupportedOperationException("Not implemented");
    }

    
    public double[] calc_nc(DoubleMatrix2D x, double[] y)
    {
        if (x instanceof jcp.bindings.libsvm.SparseDoubleMatrix2D) {
            return fastCalc_nc((jcp.bindings.libsvm.SparseDoubleMatrix2D)x, y);
        } else {
            return slowCalc_nc(x, y);
        }
    }

    
    public double[] calc_nc(DoubleMatrix2D xtr,
                            double[] ytr,
                            DoubleMatrix1D xtest,
                            double ytest)
    {
        throw new UnsupportedOperationException("Not implemented");
    }

    private void fastFit(jcp.bindings.libsvm.SparseDoubleMatrix2D x,
                         double[] y)
    {
        _model = svm.svm_train(_parameter, x, y);
    }

    private void slowFit(DoubleMatrix2D x,
                         double[] y)
    {
        svm_problem problem = new svm_problem();
        problem.l = y.length;
        // FIXME: This generates a dense instance matrix!
        problem.x = new svm_node[problem.l][];
        problem.y = y;

        for (int i = 0; i < problem.l; i++) {
            // FIXME: This generates a dense instance!
            problem.x[i] = convertInstance(x, i);
        }
        _model = svm.svm_train(problem, _parameter);
    }

    private double[] fastCalc_nc(jcp.bindings.libsvm.SparseDoubleMatrix2D x,
                                 double[] y)
    {
        double[] nc = new double[y.length];
        double[] probability = new double[_n_classes];

        if (DEBUG) {
            System.out.println("fastCalc_nc()");
        }
        for (int i = 0; i < nc.length; i++) {
            SparseDoubleMatrix1D instance = x.getRow(i);
            svm.svm_predict_probability(_model, instance, probability);

            nc[i] = probability[_class_index.get(y[i])];
            if (DEBUG) {
                System.out.println("  instance " + i + " target " + y[i] + ": " +
                                   nc[i]);
            }
        }
        return nc;
    }

    private double[] slowCalc_nc(DoubleMatrix2D x, double[] y)
    {
        double[] nc = new double[y.length];
        double[] probability = new double[_n_classes];

        for (int i = 0; i < nc.length; i++) {
            svm_node[] instance = convertInstance(x, i);
            svm.svm_predict_probability(_model, instance, probability);

            nc[i] = 1 - probability[_class_index.get(y[i])];
        }
        return nc;
    }

    private svm_node[] convertInstance(DoubleMatrix2D instances,
                                       int instance)
    {
        // FIXME: This generates a dense instance!
        svm_node[] attributes = new svm_node[instances.columns()]; 
        for (int i = 0; i < instances.columns(); i++) {
            attributes[i] = new svm_node();
            attributes[i].index = i;
            attributes[i].value = instances.getQuick(instance,i);
        }
        return attributes;
    }

}
