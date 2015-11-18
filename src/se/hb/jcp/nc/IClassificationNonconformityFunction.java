// JCP - Java Conformal Prediction framework
// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
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
package se.hb.jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

/**
 * Represents an instance of a specific non-conformity function for
 * conformal classification.
 *
 * Contract for JCP use:
 * 1. The non-conformity function must be serializable, both as untrained and
 *    as trained.
 */
public interface IClassificationNonconformityFunction
    extends java.io.Serializable
{
    /**
     * Initializes this non-conformity function with the supplied data.
     *
     * @param x    the instances.
     * @param y    the targets/classes/labels of the instances.
     */
    public void fit(DoubleMatrix2D x, double[] y);

    /**
     * Returns a new non-conformity function based on the same
     * parameters as the current one initialized with the supplied
     * data.
     *
     * @param x    the instances.
     * @param y    the targets or classes of the instances.
     * @returns a new non-conformity function.
     */
    public IClassificationNonconformityFunction fitNew(DoubleMatrix2D x,
                                                       double[] y);

    /**
     * Returns a new non-conformity function based on the same
     * parameters as the current one initialized with the supplied
     * data.
     *
     * @param xtr    the training instances.
     * @param ytr    the training targets or classes of the instances.
     * @param xtest  the test instances.
     * @param ytest  the test targets/classes/labels of the instances.
     * @returns a new non-conformity function.
     */
    public IClassificationNonconformityFunction
        fitNew(DoubleMatrix2D xtr, double[] ytr,
               DoubleMatrix1D xtest, double ytest);
    // FIXME: The above method is inefficient if the underlying classifier
    //        still needs everything in one X and Y structure.

    @Deprecated
    public double[] calc_nc(DoubleMatrix2D x, double[] y);
    public double[] calc_nc(DoubleMatrix2D xtrain, double[] ytrain,
                            DoubleMatrix1D xtest, double ytest);

    /**
     * Computes the non-conformity score for the instance x with the target y.
     *
     * @param x    the instance.
     * @param y    the target/class/label.
     * @returns the non-conformity score. Large means less conforming.
     */
    public double calculateNonConformityScore(DoubleMatrix1D x, double y);

    /**
     * Returns the classifier used by this non-conformity function.
     *
     * @returns the classifier used by this non-conformity function; or null if there isn't one.
     */
    public se.hb.jcp.ml.IClassifier getClassifier();
}
