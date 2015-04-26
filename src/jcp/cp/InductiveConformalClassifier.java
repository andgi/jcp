// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.cp;

import cern.colt.matrix.ObjectMatrix1D;
import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
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
    private double[] _calibrationScores;
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

        int n = _xcal.rows();
        _calibrationScores = new double[n];
        if (!PARALLEL) {
            for (int i = 0; i < n; i++) {
                DoubleMatrix1D instance = _xcal.viewRow(i);
                _calibrationScores[i] =
                    _nc.calculateNonConformityScore(instance, _ycal[i]);
            }
        } else {
            CalculateNCScoresAction all =
                new CalculateNCScoresAction(_xcal, _ycal, _calibrationScores,
                                            0, n);
            all.start();
        }
        Arrays.sort(_calibrationScores);
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
            // TODO: Underlying model should really only have to predict once.
            double  nc_pred = _nc.calculateNonConformityScore(x, _targets[i]);
            boolean include = Util.calculateInclusion(nc_pred,
                                                      _calibrationScores,
                                                      significance);
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
            // TODO: The underlying model should really only have to predict
            //       once per instance.
            double ncScore = _nc.calculateNonConformityScore(x, _targets[i]);
            double pValue  = Util.calculatePValue(ncScore,
                                                  _calibrationScores);
            pValues.set(i, pValue);
        }
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        oos.writeObject(_nc);
        oos.writeObject(_calibrationScores);
        oos.writeObject(_targets);
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        _nc = (IClassificationNonconformityFunction)ois.readObject();
        _calibrationScores = (double[])ois.readObject();
        _targets = (double[])ois.readObject();
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

    class CalculateNCScoresAction extends jcp.util.ParallelizedAction
    {
        DoubleMatrix2D _x;
        double[] _y;
        double[] _nonConformityScores;

        public CalculateNCScoresAction(DoubleMatrix2D x,
                                       double[] y,
                                       double[] nonConformityScores,
                                       int first, int last)
        {
            super(first, last);
            _x = x;
            _y = y;
            _nonConformityScores = nonConformityScores;
        }

        protected void compute(int i)
        {
            DoubleMatrix1D instance = _x.viewRow(i);
            _nonConformityScores[i] =
                _nc.calculateNonConformityScore(instance, _y[i]);
        }

        protected ParallelizedAction createSubtask(int first, int last)
        {
            return new CalculateNCScoresAction(_x, _y, _nonConformityScores,
                                               first, last);
        }
    }
}
