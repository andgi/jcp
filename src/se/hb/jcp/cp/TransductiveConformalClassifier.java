// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package se.hb.jcp.cp;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.ObjectMatrix1D;
import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.DenseObjectMatrix1D;
import cern.colt.matrix.impl.DenseObjectMatrix2D;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;

import se.hb.jcp.nc.IClassificationNonconformityFunction;
import se.hb.jcp.util.ParallelizedAction;

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
     * Computes the set of labels predicted for the instance x at the
     * selected significance level using prepared buffers for the training set.
     *
     * @param x             the instance.
     * @param significance  the significance level [0-1].
     * @param labels        an initialized <tt>ObjectMatrix1D</tt> to store the predicted labels.
     * @param xtr           an initialized <tt>DoubleMatrix2D</tt> containing the training instances and, last, one free slot.
     * @param ytr           an initialized <tt>double[]</tt> array containing the training instances and, last, one free slot.
     */
    private void predict(DoubleMatrix1D x, double significance,
                         ObjectMatrix1D labels,
                         DoubleMatrix2D xtr, double[] ytr)
    {
        for (int i = 0; i < _targets.length; i++) {
            // Set up the training set for this prediction.
            int last = xtr.rows() - 1;
            xtr.viewRow(last).assign(x);
            ytr[last] = _targets[i];

            // Create a nonconformity function instance and predict.
            IClassificationNonconformityFunction ncf =
                _nc.fitNew(xtr, ytr);
            double[] nc = ncf.calc_nc(xtr, ytr);
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

    /**
     * Computes the predicted p-values for the instance x
     * using prepared buffers for the training set.
     *
     * @param x        the instance.
     * @param pValues  an initialized <tt>DoubleMatrix1D</tt> to store the p-values.
     * @param xtr      an initialized <tt>DoubleMatrix2D</tt> containing the training instances and, last, one free slot.
     * @param ytr      an initialized <tt>double[]</tt> array containing the training instances and, last, one free slot.
     */
    private void predictPValues(DoubleMatrix1D x,
                                DoubleMatrix1D pValues,
                                DoubleMatrix2D xtr, double[] ytr)
    {
        for (int i = 0; i < _targets.length; i++) {
            // Set up the training set for this prediction.
            int last = xtr.rows() - 1;
            xtr.viewRow(last).assign(x);
            ytr[last] = _targets[i];

            // Create a nonconformity function instance and predict.
            IClassificationNonconformityFunction ncf =
                _nc.fitNew(xtr, ytr);
            double[] nc = ncf.calc_nc(xtr, ytr);
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

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        // Save the non-conformity function with its untrained or
        // partially(?) trained  predictor.
        // It is assumed that the predictor saves its configuration parameters.
        oos.writeObject(_nc);
        // Save the targets.
        oos.writeObject(_targets);
        // Save the training set in a space efficient representation.
        // FIXME: What representation is space efficient?
        //        Colt SparseDoubleMatrix2D isn't.
        // FIXME: The training set is currently always loaded back into the
        //        classifier's preferred representation.
        DoubleMatrix2D tmp_xtr;
        if (_xtr instanceof se.hb.jcp.bindings.jlibsvm.SparseDoubleMatrix2D) {
            tmp_xtr = _xtr;
        } else {
            tmp_xtr =
                new se.hb.jcp.bindings.jlibsvm.SparseDoubleMatrix2D
                        (_xtr.rows(),
                         _xtr.columns());
            tmp_xtr.assign(_xtr);
        }
        oos.writeObject(tmp_xtr);
        oos.writeObject(_ytr);
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        _nc = (IClassificationNonconformityFunction)ois.readObject();
        _targets = (double[])ois.readObject();
        DoubleMatrix2D tmp_xtr =
            (DoubleMatrix2D)ois.readObject();
        if(_nc != null && _nc.getClassifier() != null) {
            _xtr = _nc.getClassifier().nativeStorageTemplate().
                       like2D(tmp_xtr.rows(), tmp_xtr.columns());
            _xtr.assign(tmp_xtr);
        } else {
            _xtr = tmp_xtr;
        }
        _ytr = (double[])ois.readObject();
    }

    abstract class ClassifyAction extends se.hb.jcp.util.ParallelizedAction
    {
        protected DoubleMatrix2D _x;
        protected DoubleMatrix2D _myXtr;
        protected double[]       _myYtr;

        public ClassifyAction(DoubleMatrix2D x,
                              int first, int last)
        {
            super(first, last);
            _x = x;
        }

        protected void initialize(int first, int last)
        {
            super.initialize(first, last);
            // Create a local copy of the training set with one free slot
            // for the instance to be predicted.
            int n = _xtr.rows();
            _myXtr = _xtr.like(n + 1, _x.columns());
            _myYtr = new double[n + 1];
            // FIXME: This way to copy the data is probably very inefficient.
            //        Most of the underlying data-structures are row-oriented
            //        and this should be used to share the row data.
            for (int r = 0; r < n; r++) {
                _myXtr.viewRow(r).assign(_xtr.viewRow(r));
                _myYtr[r] = _ytr[r];
            }
        }

        protected void finalize(int first, int last)
        {
            super.finalize(first, last);
            // Allow faster reclamation.
            _myXtr = null;
            _myYtr = null;
        }
    }

    class ClassifyLabelsAction extends ClassifyAction
    {
        ObjectMatrix2D _response;
        double _significance;

        public ClassifyLabelsAction(DoubleMatrix2D x,
                                    ObjectMatrix2D response,
                                    double significance,
                                    int first, int last)
        {
            super(x, first, last);
            _response = response;
            _significance = significance;
        }

        protected void compute(int i)
        {
            DoubleMatrix1D instance = _x.viewRow(i);
            ObjectMatrix1D labels   = _response.viewRow(i);
            predict(instance, _significance, labels, _myXtr, _myYtr);
        }

        protected ParallelizedAction createSubtask(int first, int last)
        {
            return new ClassifyLabelsAction(_x, _response, _significance,
                                            first, last);
        }
    }

    class ClassifyPValuesAction extends ClassifyAction
    {
        DoubleMatrix2D _response;

        public ClassifyPValuesAction(DoubleMatrix2D x,
                                     DoubleMatrix2D response,
                                     int first, int last)
        {
            super(x, first, last);
            _response = response;
        }

        protected void compute(int i)
        {
            DoubleMatrix1D instance = _x.viewRow(i);
            DoubleMatrix1D pValues  = _response.viewRow(i);
            predictPValues(instance, pValues, _myXtr, _myYtr);
        }

        protected ParallelizedAction createSubtask(int first, int last)
        {
            return new ClassifyPValuesAction(_x, _response,
                                             first, last);
        }
    }
}
