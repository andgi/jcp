// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2015  Anders Gidenstam
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
package se.hb.jcp.nc;

import java.util.Map;
import java.util.TreeMap;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import se.hb.jcp.ml.IClassifier;
import se.hb.jcp.ml.IClassProbabilityClassifier;
import se.hb.jcp.util.ParallelizedAction;

public class ClassProbabilityNonconformityFunction
    implements IClassificationNonconformityFunction,
               java.io.Serializable
{
    private static final boolean DEBUG = false;
    private static final boolean PARALLEL = true;

    IClassProbabilityClassifier _model;
    int _n_classes;
    double[] _classes;
    Map<Double, Integer> _class_index = new TreeMap<Double, Integer>();

    public ClassProbabilityNonconformityFunction(double[] classes)
    {
        this(classes, new se.hb.jcp.bindings.libsvm.SVMClassifier());
    }

    public ClassProbabilityNonconformityFunction
               (double[] classes,
                IClassProbabilityClassifier classifier)
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

    public IClassificationNonconformityFunction fitNew(DoubleMatrix2D x,
                                                       double[] y)
    {
        ClassProbabilityNonconformityFunction ncf =
            new ClassProbabilityNonconformityFunction
                    (_classes,
                     (IClassProbabilityClassifier)_model.fitNew(x, y));
        return ncf;
    }

    public IClassificationNonconformityFunction
        fitNew(DoubleMatrix2D xtr, double[] ytr,
               DoubleMatrix1D xtest, double ytest)
    {
        int n = xtr.rows();
        DoubleMatrix2D trainingX = xtr.like(n + 1, xtr.columns());
        double[]       trainingY = new double[n + 1];
        // FIXME: This way to copy the data is probably very inefficient.
        //        Most of the underlying data-structures are row-oriented
        //        and this should be used to share the row data.
        // FIXED for: libsvm
        for (int r = 0; r < n; r++) {
            trainingX.viewRow(r).assign(xtr.viewRow(r));
            trainingY[r] = ytr[r];
        }
        trainingX.viewRow(n).assign(xtest);
        trainingY[n] = ytest;

        ClassProbabilityNonconformityFunction ncf =
            new ClassProbabilityNonconformityFunction
                    (_classes,
                     (IClassProbabilityClassifier)_model.fitNew(trainingX,
                                                                trainingY));
        return ncf;
    }

    @Deprecated
    public double[] calc_nc(DoubleMatrix2D x, double[] y)
    {
        double[] nc = new double[y.length];

        if (DEBUG) {
            System.out.println("fastCalc_nc()");
        }

        if (!PARALLEL) {
            for (int i = 0; i < nc.length; i++) {
                DoubleMatrix1D instance = x.viewRow(i);
                nc[i] = calculateNonConformityScore(instance, y[i]);

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
        double[] nc = new double[ytr.length + 1];
        double[] nctr = calc_nc(xtr, ytr);
        for (int i = 0; i < nctr.length; i++) {
            nc[i] = nctr[i];
        }
        nc[nctr.length] = calculateNonConformityScore(xtest, ytest);
        return nc;
    }

    public double calculateNonConformityScore(DoubleMatrix1D x, double y)
    {
        double[] probability = new double[_n_classes];
        _model.predict(x, probability);

        double nc = 1.0 - probability[_class_index.get(y)];
        if (DEBUG) {
            System.out.println("  instance (" + x + ") target " + y +
                               ": " + nc);
        }
        double label = _model.predict(x);
        if (probability[_class_index.get(label)] <
            probability[_classes.length - 1 - _class_index.get(label)]) {
            System.out.println("Warning! Poor model prediction (" +
                               label + ") - model label probability (" +
                               probability[_class_index.get(label)] +
                               ") match!");
        }

        return nc;
    }

    public se.hb.jcp.ml.IClassifier getClassifier()
    {
        return _model;
    }

    class CalcNCAction extends se.hb.jcp.util.ParallelizedAction
    {
        DoubleMatrix2D _x;
        double[] _y;
        double[] _nc;
        double[] _probability;

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

        protected void initialize(int first, int last)
        {
            _probability = new double[_n_classes];
        }

        protected void finalize(int first, int last)
        {
            _probability = null;
        }

        protected void compute(int i)
        {
            DoubleMatrix1D instance = _x.viewRow(i);
            _model.predict(instance, _probability);
            _nc[i] = 1.0 - _probability[_class_index.get(_y[i])];
        }

        protected ParallelizedAction createSubtask(int first, int last)
        {
            return new CalcNCAction(_x, _y, _nc, first, last);
        }
    }
}
