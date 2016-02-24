// JCP - Java Conformal Prediction framework
// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015 - 2016  Anders Gidenstam
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
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Map;
import java.util.TreeMap;

import se.hb.jcp.nc.IClassificationNonconformityFunction;
import se.hb.jcp.util.ParallelizedAction;

public class TransductiveConformalClassifier
    implements IConformalClassifier, java.io.Serializable
{
    private static final boolean PARALLEL = true;

    public IClassificationNonconformityFunction _nc;
    private double[] _classes;
    private Map<Double, Integer> _classIndex;
    private boolean _useLabelConditionalCP;

    private DoubleMatrix2D _xtr;   
    private double[] _ytr;

    public TransductiveConformalClassifier(double[] targets)
    {
        this(targets, false);
    }

    public TransductiveConformalClassifier(double[] targets,
                                           boolean  useLabelConditionalCP)
    {
        _useLabelConditionalCP = useLabelConditionalCP;
        _classes = targets;
        _classIndex = new TreeMap<Double, Integer>();
        for (int c = 0; c < _classes.length; c++) {
            _classIndex.put(_classes[c], c);
        }
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
        ObjectMatrix2D response = new DenseObjectMatrix2D(n, _classes.length);

        if (!PARALLEL) {
            // Create a local copy of the training set with one free slot
            // for the instance to be predicted.
            SimpleImmutableEntry<DoubleMatrix2D, double[]> mytr =
                 createLocalTrainingSet();
            DoubleMatrix2D myXtr = mytr.getKey();
            double[] myYtr = mytr.getValue();

            for (int i = 0; i < n; i++) {
                DoubleMatrix1D instance = x.viewRow(i);
                ObjectMatrix1D labels   = response.viewRow(i);
                predict(instance, significance, labels, myXtr, myYtr);
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
        ObjectMatrix1D response = new DenseObjectMatrix1D(_classes.length);
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
        // FIXME: This creates a whole new (n+1)-sized copy of the training
        //        set which is rather inefficient for a single prediction.
        // FIXME: Add special handling for nonconformity functions that
        //        can be trained incrementally, i.e. without retraining from
        //        the whole (n+1)-sized training set.

        // Create a local copy of the training set with one free slot
        // for the instance to be predicted.
        SimpleImmutableEntry<DoubleMatrix2D, double[]> mytr =
            createLocalTrainingSet();
        DoubleMatrix2D myXtr = mytr.getKey();
        double[] myYtr = mytr.getValue();
        predict(x, significance, labels, myXtr, myYtr);
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
        // FIXME: Parallelize over targets and/or calc_nc too?
        // Set up the training set for this prediction.
        int last = xtr.rows() - 1;
        xtr.viewRow(last).assign(x);
        for (int i = 0; i < _classes.length; i++) {
            // Set up the target for this prediction.
            ytr[last] = _classes[i];

            // Create a nonconformity function instance and predict.
            SimpleImmutableEntry<Double, double[]> ncScores =
              calculateNonConformityScore(xtr, ytr, _useLabelConditionalCP);
            boolean include  = Util.calculateInclusion(ncScores.getKey(),
                                                       ncScores.getValue(),
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
        DoubleMatrix2D response = new DenseDoubleMatrix2D(n, _classes.length);
        if (!PARALLEL) {
            // Create a local copy of the training set with one free slot
            // for the instance to be predicted.
            SimpleImmutableEntry<DoubleMatrix2D, double[]> mytr =
                 createLocalTrainingSet();
            DoubleMatrix2D myXtr = mytr.getKey();
            double[] myYtr = mytr.getValue();

            for (int i = 0; i < n; i++) {
                DoubleMatrix1D instance = x.viewRow(i);
                DoubleMatrix1D pValues  = response.viewRow(i);
                predictPValues(instance, pValues, myXtr, myYtr);
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
        DoubleMatrix1D response = new DenseDoubleMatrix1D(_classes.length);
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
        // FIXME: This creates a whole new (n+1)-sized copy of the training
        //        set which is rather inefficient for a single prediction.
        // FIXME: Add special handling for nonconformity functions that
        //        can be trained incrementally, i.e. without retraining from
        //        the whole (n+1)-sized training set.

        // Create a local copy of the training set with one free slot
        // for the instance to be predicted.
        SimpleImmutableEntry<DoubleMatrix2D, double[]> mytr =
            createLocalTrainingSet();
        DoubleMatrix2D myXtr = mytr.getKey();
        double[] myYtr = mytr.getValue();
        predictPValues(x, pValues, myXtr, myYtr);
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
        // FIXME: Parallelize over targets and/or calc_nc too?
        // Set up the training set for this prediction.
        int last = xtr.rows() - 1;
        xtr.viewRow(last).assign(x);
        for (int i = 0; i < _classes.length; i++) {
            // Set up the target for this prediction.
            ytr[last] = _classes[i];

            // Create a nonconformity function instance and predict.
            SimpleImmutableEntry<Double, double[]> ncScores =
                calculateNonConformityScore(xtr, ytr, _useLabelConditionalCP);
            double pValue  = Util.calculatePValue(ncScores.getKey(),
                                                  ncScores.getValue());
            pValues.set(i, pValue);
        }
    }

    public IClassificationNonconformityFunction getNonconformityFunction()
    {
        return _nc;
    }

    /**
     * Computes the non-conformity score of the last instance in xtr and ytr
     * using the rest of xtr and ytr as the calibration set.
     *
     * @param xtr      an <tt>DoubleMatrix2D</tt> containing the training instances and, last, the test instance.
     * @param ytr      an <tt>double[]</tt> array containing the training instances and, last, the assumed label of the test instance.
     * @param useLabelConditionalCP a <tt>boolean</tt> indicating whether label conditional conformal classification should be used.
     * @return a pair of the test instance's nonconformity score and an <tt>double[]</tt> array containing the sorted nonconformity scores of the calibration set.
     */
    private SimpleImmutableEntry<Double, double[]>
        calculateNonConformityScore(DoubleMatrix2D xtr,
                                    double[]       ytr,
                                    boolean        useLabelConditionalCP)
    {
        // Create a nonconformity function instance and compute the
        // nonconformity scores for the instance and the calibration set.
        IClassificationNonconformityFunction ncf =
            _nc.fitNew(xtr, ytr);

        double[] nc = ncf.calc_nc(xtr, ytr);
        double ncScore = nc[nc.length - 1];
        double[] ncCalibrationScores;
        if (useLabelConditionalCP) {
            ncCalibrationScores = new double[nc.length - 1];
            int c = 0;
            double target = ytr[ytr.length-1];
            for (int i = 0; i < nc.length - 1; i++) {
                if (ytr[i] == target) {
                    ncCalibrationScores[c++] = nc[i];
                }
            }
            ncCalibrationScores = Arrays.copyOf(ncCalibrationScores, c);
        } else {
            ncCalibrationScores = Arrays.copyOf(nc, nc.length - 1);
        }
        Arrays.sort(ncCalibrationScores);
        return new SimpleImmutableEntry<Double, double[]>(ncScore,
                                                          ncCalibrationScores);
    }

    /**
     * Creates a local (n+1)-sized copy of the training set.
     *
     * @return a pair containing a <tt>DoubleMatrix2D</tt> containing the training instances and, last, a single slot for the test instance; and a <tt>double[]</tt> array containing the labels of the training set and, last, a single slot for the test instance.
     */
    private
        SimpleImmutableEntry<DoubleMatrix2D, double[]> createLocalTrainingSet()
    {
        // Create a local copy of the training set with one free slot
        // for the instance to be predicted.
        int n = _xtr.rows();
        DoubleMatrix2D myXtr = _xtr.like(n + 1, _xtr.columns());
        double[] myYtr = new double[n + 1];
        // FIXME: This way to copy the data is probably very inefficient.
        //        Most of the underlying data-structures are row-oriented
        //        and this should be used to share the row data.
        // FIXED for: libsvm.
        for (int r = 0; r < n; r++) {
            myXtr.viewRow(r).assign(_xtr.viewRow(r));
            myYtr[r] = _ytr[r];
        }
        return new SimpleImmutableEntry<DoubleMatrix2D, double[]>(myXtr, myYtr);
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        // Save the non-conformity function with its untrained or
        // partially(?) trained  predictor.
        // It is assumed that the predictor saves its configuration parameters.
        oos.writeObject(_nc);
        // Save the targets.
        oos.writeObject(_classes);
        oos.writeObject(_classIndex);
        oos.writeObject(_useLabelConditionalCP);
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

    @SuppressWarnings("unchecked") // There is not much to do if the saved
                                   // value doesn't match the expected type.
    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        _nc = (IClassificationNonconformityFunction)ois.readObject();
        _classes = (double[])ois.readObject();
        _classIndex = (Map<Double, Integer>)ois.readObject();
        _useLabelConditionalCP = (boolean)ois.readObject();
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
            SimpleImmutableEntry<DoubleMatrix2D, double[]> mytr =
                 createLocalTrainingSet();
            _myXtr = mytr.getKey();
            _myYtr = mytr.getValue();
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
