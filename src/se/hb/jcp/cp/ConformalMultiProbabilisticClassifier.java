// JCP - Java Conformal Prediction framework
// Copyright (C) 2018  Anders Gidenstam
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

import java.util.SortedSet;
import java.util.TreeSet;

import se.hb.jcp.nc.IClassificationNonconformityFunction;
import se.hb.jcp.util.RealIndexedMatrix2D;

/**
 * Represents a multi-probabilistic conformal classifier with bivariate isotonic regression.
 * See [C. Zhou, "Conformal and Venn Predictors for Multi-probabilistic
 * Predictions and Their Applications", Ph.D. Thesis, Department of Computer
 * Science, Royal Holloway, University of London, 2015] for the details.
 * NOTE: The description in the Thesis above is not completely clear so the
 *       implementation here is partially done by educated guesses. Updated
 *       proofs of the desired properties remain to be done.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class ConformalMultiProbabilisticClassifier
    implements IConformalClassifier, java.io.Serializable
{
    private static final boolean PARALLEL = false;
    private static final double RESOLUTION = 5;

    private final IConformalClassifier _classifier;
    private RealIndexedMatrix2D<Double> _calibration;

    /**
     * Creates a multi-probabilistic conformal classifier with
     * bivariate isotonic regression using the supplied information.
     *
     * @param classifier the trained conformal classifier to use.
     */
    public ConformalMultiProbabilisticClassifier(IConformalClassifier classifier)
    {
        if (classifier == null ||
            !classifier.isTrained()) {
            throw new UnsupportedOperationException
                          ("The conformal classifier must be trained before use.");
        }
        _classifier = classifier;
    }

    /**
     * Calibrates this multi-probabilistic conformal classifier using the supplied data.
     *
     * @param xcal          the attributes of the calibration instances.
     * @param ycal          the targets of the calibration instances.
     */
    public void calibrate(DoubleMatrix2D xcal, double[] ycal)
    {
        int n = xcal.rows();
        ConformalClassification[] calibrationScores = _classifier.predict(xcal);
        RealIndexedMatrix2D<Double> X = new RealIndexedMatrix2D<Double>();
        RealIndexedMatrix2D<Double> W = new RealIndexedMatrix2D<Double>();

        int i = 0;
        for (ConformalClassification c : calibrationScores) {
//            System.err.println("(conf, cred) = (" +
//                               c.getPointPredictionConfidence() + ", " +
//                               c.getPointPredictionCredibility() + ") y^ = " +
//                               c.getLabelPointPrediction() + ", y = " +
//                               ycal[i]);
            double conf =
                (int)(c.getPointPredictionConfidence()*RESOLUTION)/RESOLUTION;
            double cred =
                (int)(c.getPointPredictionCredibility()*RESOLUTION)/RESOLUTION;
            if (c.getLabelPointPrediction() == ycal[i]) {
                double old = 0;
                if (X.contains(conf, cred)) {
                    old = X.get(conf, cred);
                }
                X.put(conf, cred, old + 1);
            }
            double old = 0;
            if (W.contains(conf, cred)) {
                old = W.get(conf, cred);
            }
            W.put(conf, cred, old + 1);
            i++;
        }
        _calibration = makeBivariateIsotonicArray(X, W);
        System.err.println(_calibration);
    }

    /**
     * Makes a prediction for each instance in x.
     * The method is parallellized over the instances.
     *
     * @param x             the instances.
     * @return an array containing a <tt>ConformalMultiProbabilisticClassification</tt> for each instance.
     */
    @Override
    public ConformalMultiProbabilisticClassification[] predict(DoubleMatrix2D x)
    {
        int n = x.rows();
        ConformalMultiProbabilisticClassification[] predictions
            = new ConformalMultiProbabilisticClassification[n];
        if (!PARALLEL) {
            for (int i = 0; i < n; i++) {
                DoubleMatrix1D instance = x.viewRow(i);
                predictions[i] = predict(instance);
            }
        } else {
            // FIXME!
        }
        return predictions;
    }

    /**
     * Makes a prediction for the instance x.
     *
     * @param x             the instance.
     * @return a prediction in the form of a <tt>ConformalMultiProbabilisticClassification</tt>.
     */
    @Override
    public ConformalMultiProbabilisticClassification predict(DoubleMatrix1D x)
    {
        ConformalClassification y = _classifier.predict(x);
        // FIXME: Is the below correct?
        Double pLower = _calibration.getLower(y.getPointPredictionConfidence(),
                                              y.getPointPredictionCredibility());
        Double pUpper = _calibration.getUpper(y.getPointPredictionConfidence(),
                                              y.getPointPredictionCredibility());

        if (pLower == null || pUpper == null || pLower > pUpper) {
            System.err.print(" Prediction: y^ = " + y.getLabelPointPrediction() +
                             ", p = (" + pLower + ", " + pUpper + ")");
            System.err.print(" BAD ");
            System.err.println();
        }
        return new ConformalMultiProbabilisticClassification(this,
                                                             y.getPValues(),
                                                             pLower != null ? pLower : 0.0,
                                                             pUpper != null ? pUpper : 1.0);
    }

    @Override
    public DoubleMatrix2D predictPValues(DoubleMatrix2D x)
    {
        return _classifier.predictPValues(x);
    }

    @Override
    public DoubleMatrix1D predictPValues(DoubleMatrix1D x)
    {
        return _classifier.predictPValues(x);
    }

    @Override
    public void predictPValues(DoubleMatrix1D x, DoubleMatrix1D pValues)
    {
        _classifier.predictPValues(x, pValues);
    }

    /**
     * Returns the underlying conformal classifier.
     *
     * @return the underlying conformal classifier.
     */
    public IConformalClassifier getConformalClassifier()
    {
        return _classifier;
    }

    @Override
    public IClassificationNonconformityFunction getNonconformityFunction()
    {
        return _classifier.getNonconformityFunction();
    }

    /**
     * Returns whether this classifier has been trained and calibrated.
     *
     * @return <tt>true</tt> if the classifier has been trained and calibrated or <tt>false</tt> otherwise.
     */
    @Override
    public boolean isTrained()
    {
        return _calibration != null;
    }

    @Override
    public int getAttributeCount()
    {
        return getConformalClassifier().getAttributeCount();
    }

    @Override
    public Double[] getLabels()
    {
        return getConformalClassifier().getLabels();
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate()
    {
        return getConformalClassifier().nativeStorageTemplate();
    }

    private RealIndexedMatrix2D<Double>
        makeBivariateIsotonicArray(RealIndexedMatrix2D<Double> X,
                                   RealIndexedMatrix2D<Double> W)
    {
        RealIndexedMatrix2D<Double> result = new RealIndexedMatrix2D<Double>();
        for (double row : X.getRowIndices()) {
            for (double column : X.getColumnIndices(row)) {
                result.put(row, column, X.get(row, column)/W.get(row, column));
            }
        }
        // FIXME: Ensure the bivariate isotonic property.
        // See [R. L. Dykstra and T. Robertson, "An Algorithm for Isotonic
        // Regression for two or more independent variables",
        // The Annals of Statistics, vol. 10, no. 3, pp. 708--716, 1982.]
        boolean changed = true;
        while (changed) {
            changed = makeIsotonicRows(result) | makeIsotonicColumns(result);
        }

        return result;
    }

    private boolean makeIsotonicRows(RealIndexedMatrix2D<Double> result)
    {
        boolean changed = false;
        for (double row : result.getRowIndices()) {
            Double[] column = result.getColumnIndices(row).toArray(new Double[0]);
            for (int i = 0; i < column.length-1; i++) {
                int j = i;
                while (j >= 0 &&
                       result.get(row, column[j]) > result.get(row, column[j+1])) {
                    double sum = 0;
                    for (int k = j; k < i+1; k++) {
                        sum += result.get(row, column[k]);
                    }
                    double avg = sum/(i+2 - j);
                    for (int k = j; k < i+1; k++) {
                        result.put(row, column[k], avg);
                        changed = true;
                    }
                    j--;
                }
            }
        }
        return changed;
    }

    private boolean makeIsotonicColumns(RealIndexedMatrix2D<Double> result)
    {
        boolean changed = false;
        Double[] allRows = result.getRowIndices().toArray(new Double[0]);
        SortedSet<Double> allColumns = new TreeSet<Double>();
        for (double row : allRows) {
            allColumns.addAll(result.getColumnIndices(row));
        }
        for (double column : allColumns) {
            for (int i = 0; i < allRows.length-1; i++) {
                int j = i;
                while (j >= 0 &&
                       result.getOrDefault(allRows[j], column, 0.0) >
                       result.getOrDefault(allRows[j+1], column, 0.0)) {
                    double sum = 0;
                    for (int k = j; k < i+1; k++) {
                        sum += result.getOrDefault(allRows[k], column, 0.0);
                    }
                    double avg = sum/(i+2 - j);
                    for (int k = j; k < i+1; k++) {
                        result.put(allRows[k], column, avg);
                        changed = true;
                    }
                    j--;
                }
            }
        }
        return changed;
    }
}
