// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.cp;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.ObjectMatrix1D;
import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.DenseObjectMatrix1D;
import cern.colt.matrix.impl.DenseObjectMatrix2D;

import java.util.Arrays;

import jcp.nc.IClassificationNonconformityFunction;
import jcp.util.ParallelizedAction;

public class TransductiveConformalClassifier
    implements IConformalClassifier, java.io.Serializable
{
    private static final boolean PARALLEL = true;

    public IClassificationNonconformityFunction _nc;
    private double[] _targets;

    private DoubleMatrix2D _xtr;   
    private double[] _ytr;

    public TransductiveConformalClassifier(double[] targets)
    {
        _targets = targets;
    }

    public void fit(DoubleMatrix2D xtr, double[] ytr)
    {
        _xtr = xtr;
        _ytr = ytr;
    }

    /**
     * Computes the set of labels predicted at the selected significance
     * level for each instance in x.
     * The method is parallellized over the instances.
     *
     * @param x             the instances.
     * @param significance  the significance level [0-1].
     * @return an <tt>ObjectMatrix2D</tt> containing the predicted labels for each instance.
     */
    public ObjectMatrix2D predict(DoubleMatrix2D x, double significance)
    {
        int n = x.rows();
        ObjectMatrix2D response = new DenseObjectMatrix2D(n, _targets.length);

        if (!PARALLEL) {
            for (int i = 0; i < n; i++) {
                DoubleMatrix1D instance = x.viewRow(i);
                ObjectMatrix1D labels   = response.viewRow(i);
                predict(instance, significance, labels);
            }
        } else {
            ClassifyLabelsAction all =
                new ClassifyLabelsAction(x, response, significance, 0, n);
            all.start();
        }
        return response;
    }

    /**
     * Computes the set of labels predicted for the instance x at the
     * selected significance level.
     *
     * @param x             the instance.
     * @param significance  the significance level [0-1].
     * @return an <tt>ObjectMatrix1D</tt> containing the predicted labels.
     */
    public ObjectMatrix1D predict(DoubleMatrix1D x, double significance)
    {
        ObjectMatrix1D response = new DenseObjectMatrix1D(_targets.length);
        predict(x, significance, response);
        return response;
    }

    /**
     * Computes the set of labels predicted for the instance x at the
     * selected significance level.
     *
     * @param x             the instance.
     * @param significance  the significance level [0-1].
     * @param labels        an initialized <tt>ObjectMatrix1D</tt> to store the predicted labels.
     */
    public void predict(DoubleMatrix1D x, double significance,
                        ObjectMatrix1D labels)
    {
        for (int i = 0; i < _targets.length; i++) {
            // FIXME: Parallelize over targets and/or calc_nc too?
            IClassificationNonconformityFunction ncf =
                _nc.fitNew(_xtr, _ytr, x, _targets[i]);
            double[] nc = ncf.calc_nc(_xtr, _ytr, x, _targets[i]);
            double[] nc_cal = Arrays.copyOf(nc, nc.length - 1);
            Arrays.sort(nc_cal);
            double ncScore = nc[nc.length - 1];
            boolean include =
                Util.calculateInclusion(ncScore, nc_cal, significance);
            labels.set(i, include);
        }
    }

    /**
     * Computes the predicted p-values for each target and instance in x.
     * The method is parallellized over the instances.
     *
     * @param x             the instances.
     * @return an <tt>DoubleMatrix2D</tt> containing the predicted p-values for each instance.
     */
    public DoubleMatrix2D predictPValues(DoubleMatrix2D x)
    {
        int n = x.rows();
        DoubleMatrix2D response = new DenseDoubleMatrix2D(n, _targets.length);
        if (!PARALLEL) {
            for (int i = 0; i < n; i++) {
                DoubleMatrix1D instance = x.viewRow(i);
                DoubleMatrix1D pValues  = response.viewRow(i);
                predictPValues(instance, pValues);
            }
        } else {
            ClassifyPValuesAction all =
                new ClassifyPValuesAction(x, response, 0, n);
            all.start();
        }
        return response;
    }

    /**
     * Computes the predicted p-values for the instance x.
     *
     * @param x    the instance.
     * @return an <tt>DoubleMatrix1D</tt> containing the predicted p-values.
     */
    public DoubleMatrix1D predictPValues(DoubleMatrix1D x)
    {
        DoubleMatrix1D response = new DenseDoubleMatrix1D(_targets.length);
        predictPValues(x, response);
        return response;
    }

    /**
     * Computes the predicted p-values for the instance x.
     *
     * @param x          the instance.
     * @param pValues    an initialized <tt>DoubleMatrix1D</tt> to store the p-values.
     */
    public void predictPValues(DoubleMatrix1D x, DoubleMatrix1D pValues)
    {
        for (int i = 0; i < _targets.length; i++) {
            // FIXME: Parallelize over targets and/or calc_nc too?
            IClassificationNonconformityFunction ncf =
                _nc.fitNew(_xtr, _ytr, x, _targets[i]);
            double[] nc = ncf.calc_nc(_xtr, _ytr, x, _targets[i]);
            double[] nc_cal = Arrays.copyOf(nc, nc.length - 1);
            Arrays.sort(nc_cal);
            double ncScore = nc[nc.length - 1];
            double pValue  = Util.calculatePValue(ncScore, nc_cal);
            pValues.set(i, pValue);
        }
    }

    public IClassificationNonconformityFunction getNonconformityFunction()
    {
        return _nc;
    }

    class ClassifyLabelsAction extends jcp.util.ParallelizedAction
    {
        DoubleMatrix2D _x;
        ObjectMatrix2D _response;
        double _significance;

        public ClassifyLabelsAction(DoubleMatrix2D x,
                                    ObjectMatrix2D response,
                                    double significance,
                                    int first, int last)
        {
            super(first, last);
            _x = x;
            _response = response;
            _significance = significance;
        }

        protected void compute(int i)
        {
            DoubleMatrix1D instance = _x.viewRow(i);
            ObjectMatrix1D labels   = _response.viewRow(i);
            predict(instance, _significance, labels);
        }

        protected ParallelizedAction createSubtask(int first, int last)
        {
            return new ClassifyLabelsAction(_x, _response, _significance,
                                            first, last);
        }
    }

    class ClassifyPValuesAction extends jcp.util.ParallelizedAction
    {
        DoubleMatrix2D _x;
        DoubleMatrix2D _response;

        public ClassifyPValuesAction(DoubleMatrix2D x,
                                     DoubleMatrix2D response,
                                     int first, int last)
        {
            super(first, last);
            _x = x;
            _response = response;
        }

        protected void compute(int i)
        {
            DoubleMatrix1D instance = _x.viewRow(i);
            DoubleMatrix1D pValues  = _response.viewRow(i);
            predictPValues(instance, pValues);
        }

        protected ParallelizedAction createSubtask(int first, int last)
        {
            return new ClassifyPValuesAction(_x, _response,
                                             first, last);
        }
    }
}
