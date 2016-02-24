// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2016  Anders Gidenstam
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
 * Represents an instance of a specific machine learning classification
 * algorithm.
 *
 * Contract for JCP use:
 * 1. The Classifier must be serializable, both as untrained and as trained.
 */
public interface IClassifier
    extends java.io.Serializable
{
    /**
     * Trains this classifier using the supplied data.
     *
     * @param x             the attributes of the instances.
     * @param y             the targets of the instances.
     */
    public void fit(DoubleMatrix2D x, double[] y);

    /**
     * Trains and returns a copy of this classifier using the supplied data.
     *
     * @param x             the attributes of the instances.
     * @param y             the targets of the instances.
     * @return a new <tt>IClassifier</tt> instance trained with the supplied data and using the same algorithm and parameter settings as the parent instance.
     */
    public IClassifier fitNew(DoubleMatrix2D x, double[] y);

    /**
     * Predicts the target for the supplied instance.
     *
     * @param instance      the instance
     * @return the predicted target of the instance.
     */
    public double predict(DoubleMatrix1D instance);

    /**
     * Returns the number of attributes the classifier has been trained on.
     *
     * @return Returns the number of attributes the classifier has been trained on or <tt>-1</tt> if the classifier has not been trained.
     */    
    public int getAttributeCount();

    /**
     * Returns a value of the <tt>DoubleMatrix1D</tt> derived class that is 
     * the native storage format for the classifier.
     *
     * @return a value of the <tt>DoubleMatrix1D</tt> derived class of the native storage format for the classifier.
     */    
    public DoubleMatrix1D nativeStorageTemplate();
}
