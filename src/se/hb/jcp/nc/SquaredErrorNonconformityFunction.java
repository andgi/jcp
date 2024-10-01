// JCP - Java Conformal Prediction framework
// Copyright (C) 2024  Tom le Cam
// Based on ClassificationNonconformityFunctionFactory.java.
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
import se.hb.jcp.ml.IRegressor;

public class SquaredErrorNonconformityFunction
    implements IRegressionNonconformityFunction,  java.io.Serializable
{
    private IRegressor _regressor;
    private boolean _isTrained = false;

    public SquaredErrorNonconformityFunction(IRegressor regressor)
    {
        _regressor = regressor;
    }

    @Override
    public void fit(DoubleMatrix2D x, double[] y)
    {
        _regressor.fit(x, y);
        _isTrained = true;
    }

    @Override
    public boolean isTrained()
    {
        return _isTrained;
    }

    @Override
    public double calculateNonConformityScore(DoubleMatrix1D instance, double label)
    {
        double prediction = _regressor.predict(instance);
        double error = label - prediction;
        return error * error;
    }

    @Override
    public double predict(DoubleMatrix1D instance)
    {
        return _regressor.predict(instance);
    }

    @Override
    public int getAttributeCount()
    {
        return _regressor.getAttributeCount();
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _regressor.nativeStorageTemplate();
    }

    @Override
    public DoubleMatrix2D predict(AbstractMatrix2D x, double significance)
    {
        //TODO
        System.out.println("To implement");
        return null;
    }
}
