// JCP - Java Conformal Prediction framework
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
package se.hb.jcp.ml;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

/**
 * Adds a bogus class probability estimate to the underlying machine
 * learning classification algorithm.
 */
public class BogusClassProbabilityClassifier
    implements IClassProbabilityClassifier
{
    private IClassifier _classifier;
    private double[]    _classes;

    public BogusClassProbabilityClassifier(IClassifier classifier,
                                           double[]    classes)
    {
        _classifier = classifier;
        _classes    = classes;
    }

    /**
     * Trains this classifier using the supplied data.
     *
     * @param x             the attributes of the instances.
     * @param y             the targets of the instances.
     */
    public void fit(DoubleMatrix2D x, double[] y)
    {
        _classifier.fit(x, y);
    }

    /**
     * Trains and returns a copy of this classifier using the supplied data.
     *
     * @param x             the attributes of the instances.
     * @param y             the targets of the instances.
     * @return a new <tt>IClassifier</tt> instance trained with the supplied data and using the same algorithm and parameter settings as the parent instance.
     */
    public IClassifier fitNew(DoubleMatrix2D x, double[] y)
    {
        return new BogusClassProbabilityClassifier(_classifier.fitNew(x, y),
                                                   _classes);
    }

    /**
     * Predicts the target for the supplied instance.
     *
     * @param instance      the instance
     * @return the predicted target of the instance.
     */
    public double predict(DoubleMatrix1D instance)
    {
        return _classifier.predict(instance);
    }

    /**
     * Predicts the target probabilities for the supplied instance.
     *
     * @param instance      the instance
     * @return a <tt>double[]</tt> array with the predicted probabilities for each of target values in the order assumed by JCP.
     */
    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates)
    {
        double prediction = _classifier.predict(instance);
        // FIXME: Probability hack. Assumes 2 classes labelled -1.0 and 1.0.
        probabilityEstimates[0] = 0.5 - 0.5*prediction;
        probabilityEstimates[1] = 0.5 + 0.5*prediction;
        return prediction;
    }

   /**
     * Returns the number of attributes the classifier has been trained on.
     *
     * @return Returns the number of attributes the classifier has been trained on or <tt>-1</tt> if the classifier has not been trained.
     */
    public int getAttributeCount()
    {
        return _classifier.getAttributeCount();
    }

    /**
     * Returns a value of the <tt>DoubleMatrix1D</tt> derived class that is
     * the native storage format for the classifier.
     *
     * @return a value of the <tt>DoubleMatrix1D</tt> derived class of the native storage format for the classifier.
     */
    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _classifier.nativeStorageTemplate();
    }
}
