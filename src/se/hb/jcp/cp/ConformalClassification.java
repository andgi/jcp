// JCP - Java Conformal Prediction framework
// Copyright (C) 2016  Anders Gidenstam
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

import java.util.SortedSet;
import java.util.TreeSet;

import cern.colt.matrix.DoubleMatrix1D;

/**
 * Represents a prediction made by a conformal classifier.
 * See [V. Vovk, A. Gammerman and G. Shafer, "Algorithmic Learning in a Random
 * World", Springer, 2005] for the definitions used here.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class ConformalClassification
{
    private IConformalClassifier _source;
    private DoubleMatrix1D _pValues;

    public ConformalClassification(IConformalClassifier source,
                                   DoubleMatrix1D pValues)
    {
        _source = source;
        // FIXME: Copy by value to avoid nasty surprises?
        _pValues = pValues;
    }

    /**
     * Returns the predicted p-values.
     *
     * @return the predicted p-values.
     */
    public DoubleMatrix1D getPValues()
    {
        return _pValues;
    }

    /**
     * Region prediction at a selected significance level.
     *
     * @param significanceLevel  the selected significance level.
     * @return a <tt>SortedSet&lt;Integer&gt;</tt> containing the predicted classes.
     */
    public SortedSet<Integer> getClassSet(double significanceLevel)
    {
        TreeSet<Integer> labels = new TreeSet<Integer>();
        for (int i = 0; i < _pValues.size(); i++) {
            // FIXME: > or >= for inclusion of a label?
            if (_pValues.get(i) > significanceLevel) {
                labels.add(i);
            }
        }
        return labels;
    }

    /**
     * Region prediction at a selected significance level.
     *
     * @param significanceLevel  the selected significance level.
     * @return a <tt>SortedSet&lt;Double&gt;</tt> containing the predicted class labels.
     */
    public SortedSet<Double> getLabelSet(double significanceLevel)
    {
        TreeSet<Double> labels = new TreeSet<Double>();
        for (int i = 0; i < _pValues.size(); i++) {
            // FIXME: > or >= for inclusion of a label?
            if (_pValues.get(i) > significanceLevel) {
                labels.add(_source.getLabels()[i]);
            }
        }
        return labels;
    }

    /**
     * Returns the maximum credibility point prediction class number.
     *
     * @return the maximum credibility class number or -1 if no unique such class/label exists.
     */
    public int getClassPointPrediction()
    {
        // FIXME: Would it be better to just return the maximum p-value class or
        //        -1 if it is not unique?
        double targetSignificanceLevel = (1.0 - getPointPredictionConfidence());
        // Floating point inaccuracy makes it likely that more than one class
        // may be included at the prescribed significance level.
        // To avoid this do a binary search from above towards the limit.
        double significanceLevel = 2.0 * targetSignificanceLevel;
        while (significanceLevel > targetSignificanceLevel) {
            for (int i = 0; i < _pValues.size(); i++) {
                if (_pValues.get(i) > significanceLevel) {
                    return i;
                }
            }
            significanceLevel -=
                0.5*(significanceLevel - targetSignificanceLevel);
        }
        return -1;
    }

    /**
     * Returns the maximum credibility point prediction label.
     *
     * @return the maximum credibility class label or NaN if no unique such class/label exists.
     */
    public double getLabelPointPrediction()
    {
        int predictedClass = getClassPointPrediction();
        if (0 <= predictedClass &&
            predictedClass < _source.getLabels().length) {
            return _source.getLabels()[predictedClass];
        } else {
            return Double.NaN;
        }
    }

    /**
     * Returns the confidence of the class/label point prediction, i.e.
     * the greatest 1-significance level for which the label set is at
     * most a single value.
     *
     * @return the confidence of the class/label point prediction.
     */
    public double getPointPredictionConfidence()
    {
        double largestPValue = -1.0;
        double secondLargestPValue = -1.0;
        for (int i = 0; i < _pValues.size(); i++) {
            if (_pValues.get(i) > largestPValue) {
                secondLargestPValue = largestPValue;
                largestPValue = _pValues.get(i);
            } else if (_pValues.get(i) > secondLargestPValue) {
                secondLargestPValue = _pValues.get(i);
            }
        }
        return 1.0 - secondLargestPValue;
    }

    /**
     * Returns the credibility of the class/label point prediction, i.e.
     * the smallest significance level for which the label set is empty.
     *
     * @return the credibility of the class/label point prediction.
     */
    public double getPointPredictionCredibility()
    {
        double largestPValue = -1.0;
        for (int i = 0; i < _pValues.size(); i++) {
            if (_pValues.get(i) > largestPValue) {
                largestPValue = _pValues.get(i);
            }
        }
        return largestPValue;
    }

    /**
     * Returns the conformal classifier that made this prediction.
     *
     * @return the conformal classifier that made this prediction.
     */
    public IConformalClassifier getSource()
    {
        return _source;
    }
}
