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
package se.hb.jcp.ml;

import cern.colt.matrix.DoubleMatrix1D;

/**
 * Specifies an interface for SVM classifiers giving access to internal SVM
 * specific information.
 *
 * Contract for JCP use:
 * 1. The methods implemented for this interface must be reentrant.
 *
 */
public interface ISVMClassifier
    extends IClassifier
{
    /**
     * Returns the signed distance between the separating hyperplane and the
     * instance.
     *
     * @param instance  the instance.
     * @return the signed distance between the separating hyperplane and the instance.
     */
    public double distanceFromSeparatingPlane(DoubleMatrix1D instance);

}
