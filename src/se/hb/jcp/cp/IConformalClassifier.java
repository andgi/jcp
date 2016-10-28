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
package se.hb.jcp.cp;

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
    extends se.hb.jcp.ml.IClassifierInformation, java.io.Serializable
{
    /**
     * Makes a prediction for each instance in x.
     * The method is parallellized over the instances.
     *
     * @param x             the instances.
     * @return an array containing a <tt>ConformalClassification</tt> for each instance.
     */
    public ConformalClassification[] predict(DoubleMatrix2D x);

    /**
     * Makes a prediction for the instance x.
     *
     * @param x             the instance.
     * @return a prediction in the form of a <tt>ConformalClassification</tt>.
     */
    public ConformalClassification predict(DoubleMatrix1D x);

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
