// Copyright (C) 2014 - 2015  Anders Gidenstam
// License: to be defined.
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
     * @returns a new <tt>IClassifier</tt> instance trained with the supplied data and using the same algorithm and parameter settings as the parent instance.
     */
    public IClassifier fitNew(DoubleMatrix2D x, double[] y);

    /**
     * Predicts the target for the supplied instance.
     *
     * @param instance      the instance
     * @returns the predicted target of the instance.
     */
    public double predict(DoubleMatrix1D instance);

    /**
     * Predicts the target probabilities for the supplied instance.
     *
     * @param instance      the instance
     * @returns a <tt>double[]</tt> array with the predicted probabilities for each of target values.
     */
    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates);

    /**
     * Returns the number of attributes the classifier has been trained on.
     *
     * @returns Returns the number of attributes the classifier has been trained on or <tt>-1</tt> if the classifier has not been trained.
     */    
    public int getAttributeCount();

    /**
     * Returns a value of the <tt>DoubleMatrix1D</tt> derived class that is 
     * the native storage format for the classifier.
     *
     * @returns a value of the <tt>DoubleMatrix1D</tt> derived class of the native storage format for the classifier.
     */    
    public DoubleMatrix1D nativeStorageTemplate();
}
