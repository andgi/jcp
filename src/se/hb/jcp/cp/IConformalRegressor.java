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
package se.hb.jcp.cp;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public interface IConformalRegressor {
    void fit(DoubleMatrix2D xtr, double[] ytr, DoubleMatrix2D xcal, double[] ycal);
    double[] predictIntervals(DoubleMatrix1D x, double confidence);
    double[][] predictIntervals(DoubleMatrix2D x, double confidence);
    boolean isTrained();
    int getAttributeCount();
    DoubleMatrix1D nativeStorageTemplate();
}
