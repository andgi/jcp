// JCP - Java Conformal Prediction framework
// Copyright (C) 2016, 2019  Anders Gidenstam
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

import cern.colt.matrix.DoubleMatrix2D;

/**
 * Base class for regressors that provide implementations of some of the
 * generic IRegressorInformation methods.
 */
public abstract class RegressorBase
    implements IRegressor,
               java.io.Serializable
{
    private int _attributeCount = -1;
    private boolean _trained = false;

    @Override
    public final void fit(DoubleMatrix2D x, double[] y)
    {
        // Setup remaining regressor information.
        _attributeCount = x.columns();

        // Train the model.
        internalFit(x, y);
        _trained = true;
    }

    @Override
    public final boolean isTrained()
    {
        return _trained;
    }

    @Override
    public final int getAttributeCount()
    {
        return _attributeCount;
    }

    protected abstract void internalFit(DoubleMatrix2D x, double[] y);
}
