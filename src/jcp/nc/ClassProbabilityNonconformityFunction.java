// Copyright (C) 2014 - 2015  Anders Gidenstam
// License: to be defined.
package jcp.nc;

import java.util.Map;
import java.util.TreeMap;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import jcp.ml.IClassifier;
import jcp.util.ParallelizedAction;

public class ClassProbabilityNonconformityFunction
    implements IClassificationNonconformityFunction,
               java.io.Serializable
{
    private static final boolean DEBUG = false;
    private static final boolean PARALLEL = true;

    IClassifier _model;
    int _n_classes;
    double[] _classes;
    Map<Double, Integer> _class_index = new TreeMap<Double, Integer>();

    public ClassProbabilityNonconformityFunction(double[] classes)
    {
        this(classes, new jcp.bindings.libsvm.SVMClassifier());
    }

    public ClassProbabilityNonconformityFunction(double[] classes,
                                                 IClassifier classifier)
    {
        _n_classes = classes.length;
        _classes = classes;

        for (int i = 0; i < _n_classes; i++) {
            _class_index.put(_classes[i], i);
        }

        _model = classifier;
    }

    public void fit(DoubleMatrix2D x,
                    double[] y)
    {
        _model.fit(x, y);
    }

    
    public void fit(DoubleMatrix2D xtr,
                    double[] ytr,
                    DoubleMatrix1D xtest,
                    double ytest)
    {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Deprecated
    public double[] calc_nc(DoubleMatrix2D x, double[] y)
    {
        double[] nc = new double[y.length];
        double[] probability = new double[_n_classes];

        if (DEBUG) {
            System.out.println("fastCalc_nc()");
        }

        if (!PARALLEL) {
            for (int i = 0; i < nc.length; i++) {
                DoubleMatrix1D instance = x.viewRow(i);
                _model.predict(instance, probability);
                
                nc[i] = probability[_class_index.get(y[i])];
                if (DEBUG) {
                    System.out.println("  instance " + i + " target " + y[i] +
                                       ": " + nc[i]);
                }
            }
        } else {
            CalcNCAction all = new CalcNCAction(x, y, nc, 0, y.length);
            all.start();
        }
        return nc;
    }

    public double[] calc_nc(DoubleMatrix2D xtr,
                            double[] ytr,
                            DoubleMatrix1D xtest,
                            double ytest)
    {
        throw new UnsupportedOperationException("Not implemented");
    }

    public double calculateNonConformityScore(DoubleMatrix1D x, double y)
    {
        double[] probability = new double[_n_classes];
        _model.predict(x, probability);

        double nc = probability[_class_index.get(y)];
        if (DEBUG) {
            System.out.println("  instance (" + x + ") target " + y +
                               ": " + nc);
        }
        return nc;
    }

    public jcp.ml.IClassifier getClassifier()
    {
        return _model;
    }

    class CalcNCAction extends jcp.util.ParallelizedAction
    {
        DoubleMatrix2D _x;
        double[] _y;
        double[] _nc;

        public CalcNCAction(DoubleMatrix2D x,
                            double[] y,
                            double[] nc,
                            int first, int last)
        {
            super(first, last);
            _x = x;
            _y = y;
            _nc = nc;
        }

        protected void compute(int i)
        {
            double[] probability = new double[_n_classes];

            DoubleMatrix1D instance = _x.viewRow(i);
            _model.predict(instance, probability);
            _nc[i] = probability[_class_index.get(_y[i])];
        }

        protected ParallelizedAction createSubtask(int first, int last)
        {
            return new CalcNCAction(_x, _y, _nc, first, last);
        }
    }
}
