// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.cp;

import cern.colt.matrix.ObjectMatrix1D;
import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseObjectMatrix1D;
import cern.colt.matrix.impl.DenseObjectMatrix2D;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import java.util.Arrays;

import jcp.nc.IClassificationNonconformityFunction;
import jcp.util.ParallelizedAction;

public class InductiveConformalClassifier
    implements java.io.Serializable
{
    private static final boolean PARALLEL = true;

    public IClassificationNonconformityFunction _nc;
    private double[] _calibration_scores;
    private double[] _targets;

    private DoubleMatrix2D _xtr;
    private DoubleMatrix2D _xcal;

    private double[] _ytr;
    private double[] _ycal;

    public InductiveConformalClassifier(double[] targets)
    {
        _targets = targets;
    }

    public void fit(DoubleMatrix2D xtr, double[] ytr,
                    DoubleMatrix2D xcal, double[] ycal)
    {
        _xtr = xtr;
        _ytr = ytr;
        _xcal = xcal;
        _ycal = ycal;
        
        _nc.fit(_xtr, _ytr);
        _calibration_scores = _nc.calc_nc(_xcal, _ycal);
        Arrays.sort(_calibration_scores);
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
            ClassifyAction all =
                new ClassifyAction(x, response, significance, 0, n);
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
            // TODO: Underlying model should really only have to predict once.
            double  nc_pred = _nc.calculateNonConformityScore(x, _targets[i]);
            boolean include = Util.calculateInclusion(nc_pred,
                                                      _calibration_scores,
                                                      significance);
            labels.set(i, include);
        }
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        oos.writeObject(_nc);
        oos.writeObject(_calibration_scores);
        oos.writeObject(_targets);
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        _nc = (IClassificationNonconformityFunction)ois.readObject();
        _calibration_scores = (double[])ois.readObject();
        _targets = (double[])ois.readObject();
    }

    class ClassifyAction extends jcp.util.ParallelizedAction
    {
        DoubleMatrix2D _x;
        ObjectMatrix2D _response;
        double _significance;

        public ClassifyAction(DoubleMatrix2D x,
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
            return new ClassifyAction(_x, _response, _significance,
                                      first, last);
        }
    }
}
