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
 * Specifies a set of information that every classifier should be able to
 * provide.
 *
 * Contract for JCP use:
 * 1. The methods implemented for this interface must be reentrant.
 *
 */
public interface IClassifierInformation
    extends java.io.Serializable
{
    /**
     * Returns whether this classifier has been trained.
     *
     * @return Returns <tt>true</tt> if the classifier has been trained or <tt>false</tt> otherwise.
     */
    public boolean isTrained();

    /**
     * Returns the number of attributes the classifier has been trained on.
     *
     * @return Returns the number of attributes the classifier has been trained on or <tt>-1</tt> if the classifier has not been trained.
     */
    public int getAttributeCount();

    /**
     * Returns the set of class labels the classifier has been trained on.
     *
     * @return Returns the class labels the classifier has been trained on in increasing order (the index is the class id) or <tt>null</tt> if the classifier has not been trained.
     */
    public Double[] getLabels();

    /**
     * Returns a value of the <tt>DoubleMatrix1D</tt> derived class that is
     * the native storage format for the classifier.
     *
     * @return a value of the <tt>DoubleMatrix1D</tt> derived class of the native storage format for the classifier.
     */
    public DoubleMatrix1D nativeStorageTemplate();
}
