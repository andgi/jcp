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
import java.util.SortedMap;
import java.util.TreeMap;

import se.hb.jcp.nc.IClassificationNonconformityFunction;
import se.hb.jcp.util.ParallelizedAction;

public class InductiveConformalClassifier
    implements IConformalClassifier, java.io.Serializable
{
    private static final boolean PARALLEL = true;

    public IClassificationNonconformityFunction _nc;
    private Double[] _classes;
    private SortedMap<Double, Integer> _classIndex;
    private int _attributeCount = -1;
    // For normal conformal prediction.
    private double[] _calibrationScores;
    // For label/class-conditional conformal prediction.
    private boolean    _useLabelConditionalCP;
    private double[][] _classCalibrationScores;

    private DoubleMatrix2D _xtr;
    private DoubleMatrix2D _xcal;

    private double[] _ytr;
    private double[] _ycal;

    public InductiveConformalClassifier(double[] targets)
    {
        this(targets, false);
    }

    public InductiveConformalClassifier(double[] targets,
                                        boolean  useLabelConditionalCP)
    {
        _useLabelConditionalCP = useLabelConditionalCP;
        _classIndex = new TreeMap<Double, Integer>();
        for (int c = 0; c < targets.length; c++) {
            _classIndex.put(targets[c], c);
        }
        _classes = _classIndex.keySet().toArray(new Double[0]);
        Arrays.sort(_classes); // FIXME: Redundant?
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
        if (_useLabelConditionalCP) {
            _classCalibrationScores = new double[_classes.length][];
            for (int c = 0; c < _classes.length; c++) {
                _classCalibrationScores[c] = new double[n];
                Arrays.fill(_classCalibrationScores[c], -Double.MAX_VALUE);
            }
        }
        if (!PARALLEL) {
            for (int i = 0; i < n; i++) {
                DoubleMatrix1D instance = _xcal.viewRow(i);
                _calibrationScores[i] =
                    _nc.calculateNonConformityScore(instance, _ycal[i]);
                if (_useLabelConditionalCP) {
                    _classCalibrationScores[_classIndex.get(_ycal[i])][i] =
                        _calibrationScores[i];
                }
            }
        } else {
            CalculateNCScoresAction all =
                new CalculateNCScoresAction(_xcal, _ycal, _calibrationScores,
                                            _classCalibrationScores,
                                            0, n);
            all.start();
        }
        Arrays.sort(_calibrationScores);
        if (_useLabelConditionalCP) {
            // FIXME: This is ugly.
            for (int c = 0; c < _classes.length; c++) {
                Arrays.sort(_classCalibrationScores[c]);
                int i = 0;
                while(i < _classCalibrationScores[c].length &&
                      _classCalibrationScores[c][i] == -Double.MAX_VALUE) {
                    i++;
                }
                if (i < _classCalibrationScores[c].length) {
                    _classCalibrationScores[c] =
                        Arrays.copyOfRange(_classCalibrationScores[c],
                                           i,
                                           _classCalibrationScores[c].length);
                } else {
                    // There were no examples with this class/label.
                    _classCalibrationScores[c] = new double[0];
                }
                System.out.println("Calibration set size for class " + c +
                                   " label " + _classes[c] + " is " +
                                   _classCalibrationScores[c].length);
            }
        }
        _attributeCount = xtr.columns();
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
        for (int i = 0; i < _classes.length; i++) {
            // FIXME: Underlying model should really only have to predict once.
            double  nc_pred = _nc.calculateNonConformityScore(x, _classes[i]);
            boolean include;
            if (_useLabelConditionalCP) {
                include = Util.calculateInclusion(nc_pred,
                                                  _classCalibrationScores[i],
                                                  significance);
            } else {
                include = Util.calculateInclusion(nc_pred,
                                                  _calibrationScores,
                                                  significance);
            }
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
        for (int i = 0; i < _classes.length; i++) {
            // TODO: The underlying model should really only have to predict
            //       once per instance.
            double ncScore = _nc.calculateNonConformityScore(x, _classes[i]);
            double pValue;
            if (_useLabelConditionalCP) {
                pValue = Util.calculatePValue(ncScore,
                                              _classCalibrationScores[i]);
            } else {
                pValue = Util.calculatePValue(ncScore,
                                              _calibrationScores);
            }
            pValues.set(i, pValue);
        }
    }

    public IClassificationNonconformityFunction getNonconformityFunction()
    {
        return _nc;
    }

    public boolean isTrained()
    {
        return getAttributeCount() >= 0;
    }

    public int getAttributeCount()
    {
        return _attributeCount;
    }

    public Double[] getLabels()
    {
        return _classes;
    }

    public DoubleMatrix1D nativeStorageTemplate()
    {
        if (getNonconformityFunction() != null &&
            getNonconformityFunction().getClassifier() != null) {
            return getNonconformityFunction().getClassifier()
                       .nativeStorageTemplate();
        } else {
            return new cern.colt.matrix.impl.SparseDoubleMatrix1D(0);
        }
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        oos.writeObject(_nc);
        oos.writeObject(_classes);
        oos.writeObject(_classIndex);
        oos.writeObject(_attributeCount);
        oos.writeObject(_calibrationScores);
        oos.writeObject(_useLabelConditionalCP);
        oos.writeObject(_classCalibrationScores);
    }

    @SuppressWarnings("unchecked") // There is not much to do if the saved
                                   // value doesn't match the expected type.
    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        _nc = (IClassificationNonconformityFunction)ois.readObject();
        _classes = (Double[])ois.readObject();
        _classIndex = (SortedMap<Double, Integer>)ois.readObject();
        _attributeCount = (int)ois.readObject();
        _calibrationScores = (double[])ois.readObject();
        _useLabelConditionalCP = (boolean)ois.readObject();
        _classCalibrationScores = (double[][])ois.readObject();
    }

    class ClassifyLabelsAction extends se.hb.jcp.util.ParallelizedAction
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

    class ClassifyPValuesAction extends se.hb.jcp.util.ParallelizedAction
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

    class CalculateNCScoresAction extends se.hb.jcp.util.ParallelizedAction
    {
        DoubleMatrix2D _x;
        double[] _y;
        double[] _nonConformityScores;
        double[][] _classNonConformityScores;

        public CalculateNCScoresAction(DoubleMatrix2D x,
                                       double[]       y,
                                       double[]       nonConformityScores,
                                       double[][]     classNonConformityScores,
                                       int first, int last)
        {
            super(first, last);
            _x = x;
            _y = y;
            _nonConformityScores = nonConformityScores;
            _classNonConformityScores = classNonConformityScores;
        }

        protected void compute(int i)
        {
            DoubleMatrix1D instance = _x.viewRow(i);
            _nonConformityScores[i] =
                _nc.calculateNonConformityScore(instance, _y[i]);
            if (_classNonConformityScores != null) {
                _classNonConformityScores[_classIndex.get(_y[i])][i] =
                    _nonConformityScores[i];
            }
        }

        protected ParallelizedAction createSubtask(int first, int last)
        {
            return new CalculateNCScoresAction(_x, _y, _nonConformityScores,
                                               _classNonConformityScores,
                                               first, last);
        }
    }
}
