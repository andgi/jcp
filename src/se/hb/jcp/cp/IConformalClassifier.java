// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package se.hb.jcp.cp;

import cern.colt.matrix.ObjectMatrix1D;
import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import se.hb.jcp.nc.IClassificationNonconformityFunction;

/**
 * Represents an instance of a specific conformal classification
 * algorithm.
 *
 * Contract for JCP use:
 * 1. The ConformalClassifier must be serializable, both as untrained and as
 *    trained.
 */
public interface IConformalClassifier
    extends java.io.Serializable
{
    /**
     * Computes the set of labels predicted at the selected significance
     * level for each instance in x.
     * The method is parallellized over the instances.
     *
     * @param x             the instances.
     * @param significance  the significance level [0-1].
     * @return an <tt>ObjectMatrix2D</tt> containing the predicted labels for each instance.
     */
    public ObjectMatrix2D predict(DoubleMatrix2D x, double significance);

    /**
     * Computes the set of labels predicted for the instance x at the
     * selected significance level.
     *
     * @param x             the instance.
     * @param significance  the significance level [0-1].
     * @return an <tt>ObjectMatrix1D</tt> containing the predicted labels.
     */
    public ObjectMatrix1D predict(DoubleMatrix1D x, double significance);

    /**
     * Computes the set of labels predicted for the instance x at the
     * selected significance level.
     *
     * @param x             the instance.
     * @param significance  the significance level [0-1].
     * @param labels        an initialized <tt>ObjectMatrix1D</tt> to store the predicted labels.
     */
    public void predict(DoubleMatrix1D x, double significance,
                        ObjectMatrix1D labels);

    /**
     * Computes the predicted p-values for each target and instance in x.
     * The method is parallellized over the instances.
     *
     * @param x             the instances.
     * @return an <tt>DoubleMatrix2D</tt> containing the predicted p-values for each instance.
     */
    public DoubleMatrix2D predictPValues(DoubleMatrix2D x);

    /**
     * Computes the predicted p-values for the instance x.
     *
     * @param x    the instance.
     * @return an <tt>DoubleMatrix1D</tt> containing the predicted p-values.
     */
    public DoubleMatrix1D predictPValues(DoubleMatrix1D x);

    /**
     * Computes the predicted p-values for the instance x.
     *
     * @param x          the instance.
     * @param pValues    an initialized <tt>DoubleMatrix1D</tt> to store the p-values.
     */
    public void predictPValues(DoubleMatrix1D x, DoubleMatrix1D pValues);

    /**
     * Returns the associated nonconformity function.
     *
     * @return the associated <tt>IClassificationNonconformityFunction</tt>.
     */
    public IClassificationNonconformityFunction getNonconformityFunction();

}
