// JCP - Java Conformal Prediction framework
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
package se.hb.jcp.bindings.jliblinear;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.File;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import de.bwaldvogel.liblinear.*;

import se.hb.jcp.ml.RegressorBase;
import se.hb.jcp.ml.IRegressor;

public class LinearRegressor
    extends RegressorBase
    implements java.io.Serializable
{
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected JSONObject _jsonParameters;
    protected Model _model;

    public LinearRegressor()
    {
    }

    public LinearRegressor(JSONObject parameters)
    {
        this();
        _jsonParameters = parameters;
    }

    protected void internalFit(DoubleMatrix2D x, double[] y)
    {
        Parameter parameters = readParameters();

        SparseDoubleMatrix2D tmp_x;
        if (x instanceof SparseDoubleMatrix2D) {
            tmp_x = (SparseDoubleMatrix2D)x;
            for (int r = 0; r < tmp_x.rows(); ++r) {
                tmp_x.viewRow(r).cardinality();
            }
        } else {
            tmp_x = new SparseDoubleMatrix2D(x.rows(), x.columns());
            tmp_x.assign(x);
        }
        Problem problem = new Problem();
        problem.bias = 0.0;
        problem.l = y.length;
        problem.n = tmp_x.columns();
        problem.x = tmp_x._rows;
        problem.y = y;

        _model = Linear.train(problem, parameters);
    }

    public IRegressor fitNew(DoubleMatrix2D x, double[] y)
    {
        LinearRegressor clone = new LinearRegressor(_jsonParameters);
        clone.fit(x, y);
        return clone;
    }

    public double predict(DoubleMatrix1D instance)
    {
        SparseDoubleMatrix1D tmp_instance;
        if (instance instanceof SparseDoubleMatrix1D) {
            tmp_instance = (SparseDoubleMatrix1D)instance;
        } else {
            tmp_instance = new SparseDoubleMatrix1D(instance.size());
            tmp_instance.assign(instance);
        }

        return Linear.predict(_model, tmp_instance._nodes);
    }

    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }

    private Parameter readParameters()
    {
        Parameter parameters = new Parameter(SolverType.L2R_L2LOSS_SVR, 1.0, 0.01);

        if (_jsonParameters != null) {
            if (_jsonParameters.has("solver_type")) {
                String type = _jsonParameters.getString("solver_type");
                if (type.equals("L2R_L2LOSS_SVR")) {
                    parameters.setSolverType(SolverType.L2R_L2LOSS_SVR);
                } else if (type.equals("L2R_L1LOSS_SVR_DUAL")) {
                    parameters.setSolverType(SolverType.L2R_L1LOSS_SVR_DUAL);
                } else if (type.equals("L2R_L2LOSS_SVR_DUAL")) {
                    parameters.setSolverType(SolverType.L2R_L2LOSS_SVR_DUAL);
                } else {
                    throw new IllegalArgumentException("Unknown SolverType '" + type + "'.");
                }
            }
            if (_jsonParameters.has("C")) {
                parameters.setC(_jsonParameters.getDouble("C"));
            }
            if (_jsonParameters.has("eps")) {
                parameters.setEps(_jsonParameters.getDouble("eps"));
            }
            if (_jsonParameters.has("p")) {
                parameters.setP(_jsonParameters.getDouble("p"));
            }
        }
        return parameters;
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        if (_jsonParameters != null) {
            oos.writeObject(_jsonParameters.toString());
        } else {
            oos.writeObject(null);
        }
        if (_model != null) {
            String fileName =
                Long.toHexString(Double.doubleToLongBits(Math.random())) +
                ".jliblinear";

            File file = new File(fileName);
            // Save the liblinear model to a separate file.
            Linear.saveModel(file, _model);
            // Save the model file name.
            oos.writeObject(fileName);
        } else {
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        String jsonText = (String)ois.readObject();
        if (jsonText != null) {
            _jsonParameters = new JSONObject(jsonText);
        }
        String fileName = (String)ois.readObject();
        if (fileName != null) {
            // Load the liblinear model from the designated file.
            File file = new File(fileName);
            // Load the model from a separate file.
            _model = Linear.loadModel(file);
        }
    }

    static {
        Linear.setDebugOutput(null);
    }
}

