// JCP - Java Conformal Prediction framework
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
public interface IClassProbabilityClassifier
    extends IClassifier, java.io.Serializable
{
    /**
     * Predicts the target probabilities for the supplied instance.
     *
     * @param instance      the instance
     * @returns a <tt>double[]</tt> array with the predicted probabilities for each of target values in the order assumed by JCP.
     */
    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates);
}
