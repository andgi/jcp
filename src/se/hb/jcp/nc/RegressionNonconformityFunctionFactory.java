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
package se.hb.jcp.nc;

import se.hb.jcp.ml.IRegressor;

/**
 * Singleton factory for JCP regression nonconformity functions.
 * 
 */
public final class RegressionNonconformityFunctionFactory {
    private static final RegressionNonconformityFunctionFactory _theInstance = new RegressionNonconformityFunctionFactory();
    private static final String[] _ncfNames = {
        "absolute error nonconformity function",
        "squared error nonconformity function"
    };

    private RegressionNonconformityFunctionFactory() {
    }

    public String[] getNonconformityFunctions() {
        return _ncfNames;
    }

    public IRegressionNonconformityFunction createNonconformityFunction(int type, IRegressor regressor) {
        switch (type) {
        case 0:
            return new AbsoluteErrorNonconformityFunction(regressor);
        case 1:
            return new SquaredErrorNonconformityFunction(regressor);
        default:
            throw new UnsupportedOperationException("Unknown nonconformity function type.");
        }
    }

    public static RegressionNonconformityFunctionFactory getInstance() {
        return _theInstance;
    }
}
