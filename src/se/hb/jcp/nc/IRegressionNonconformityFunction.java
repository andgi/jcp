// JCP - Java Conformal Prediction framework
// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2024  Tom le Cam
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
import cern.colt.matrix.impl.AbstractMatrix2D;

public interface IRegressionNonconformityFunction {
    public DoubleMatrix2D predict(AbstractMatrix2D x, double significance);
    void fit(DoubleMatrix2D x, double[] y);
    boolean isTrained();
    double calculateNonConformityScore(DoubleMatrix1D instance, double label);
    double predict(DoubleMatrix1D instance);
    int getAttributeCount();
    DoubleMatrix1D nativeStorageTemplate();
}


