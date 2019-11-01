// JCP - Java Conformal Prediction framework
// Copyright (C) 2015 - 2016, 2019  Anders Gidenstam
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

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassifier;
import se.hb.jcp.ml.IClassProbabilityClassifier;

public class LinearClassifier
    extends ClassifierBase
    implements IClassProbabilityClassifier,
               java.io.Serializable
{
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected JSONObject _jsonParameters;
    protected Model _model;
    private   boolean _switchProbabilities = false;

    public LinearClassifier()
    {
    }

    public LinearClassifier(JSONObject parameters)
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
            // FIXME: Look at each row so that the matrix is properly updated.
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
        _switchProbabilities =
            (double)_model.getLabels()[0] != (double)getLabels()[0];
        //System.out.println("jliblinear.LinearClassifier.internalFit(): " +
        //                   "_switchProbabilities = " +
        //                   (double)_model.getLabels()[0] + " != " +
        //                   (double)getLabels()[0] + " = " +
        //                   _switchProbabilities);
    }

    public IClassifier fitNew(DoubleMatrix2D x, double[] y)
    {
        LinearClassifier clone = new LinearClassifier(_jsonParameters);
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

    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates)
    {
        SparseDoubleMatrix1D tmp_instance;
        if (instance instanceof SparseDoubleMatrix1D) {
            tmp_instance = (SparseDoubleMatrix1D)instance;
        } else {
            tmp_instance = new SparseDoubleMatrix1D(instance.size());
            tmp_instance.assign(instance);
        }

        double prediction = Linear.predictProbability(_model,
                                                      tmp_instance._nodes,
                                                      probabilityEstimates);
        // jliblinear seem to use the reverse order of labels, so reverse
        // the array of probability estimates before returning them.
        // FIXME: Currently using Model.getLabels() and this.getLabels() to
        //        ensure compatibility.
        if (_switchProbabilities) {
            int i = 0;
            int j = probabilityEstimates.length-1;
            for (; i < j; i++, j--) {
                double tmp = probabilityEstimates[i];
                probabilityEstimates[i] = probabilityEstimates[j];
                probabilityEstimates[j] = tmp;
            }
        }
        return prediction;
    }

    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }

    private Parameter readParameters()
    {
        // Default parameters.
        Parameter parameters = new Parameter(SolverType.L2R_LR,
                                             1.0,
                                             0.01);

        if (_jsonParameters != null) {
            if (_jsonParameters.has("solver_type")) {
                String type = _jsonParameters.getString("solver_type");
                if (type.equals("L2R_LR")) {
                    parameters.setSolverType(SolverType.L2R_LR);
                } else if (type.equals("L2R_L2LOSS_SVC_DUAL")) {
                    parameters.setSolverType(SolverType.L2R_L2LOSS_SVC_DUAL);
                } else if (type.equals("L2R_L2LOSS_SVC")) {
                    parameters.setSolverType(SolverType.L2R_L2LOSS_SVC);
                } else if (type.equals("L2R_L1LOSS_SVC_DUAL")) {
                    parameters.setSolverType(SolverType.L2R_L1LOSS_SVC_DUAL);
                } else if (type.equals("MCSVM_CS")) {
                    parameters.setSolverType(SolverType.MCSVM_CS);
                } else if (type.equals("L1R_L2LOSS_SVC")) {
                    parameters.setSolverType(SolverType.L1R_L2LOSS_SVC);
                } else if (type.equals("L1R_LR")) {
                    parameters.setSolverType(SolverType.L1R_LR);
                } else if (type.equals("L2R_LR_DUAL")) {
                    parameters.setSolverType(SolverType.L2R_LR_DUAL);
                } else if (type.equals("L2R_L2LOSS_SVR")) {
                    parameters.setSolverType(SolverType.L2R_L2LOSS_SVR);
                } else if (type.equals("L2R_L2LOSS_SVR_DUAL")) {
                    parameters.setSolverType(SolverType.L2R_L2LOSS_SVR_DUAL);
                } else if (type.equals("L2R_L1LOSS_SVR_DUAL")) {
                    parameters.setSolverType(SolverType.L2R_L1LOSS_SVR_DUAL);
                } else {
                    throw new IllegalArgumentException
                                  ("se.hb.jcp.bindings.jliblinear.LinearClassifier: "
                                   + "Unknown SolverType '" + type + "'.");
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
        // Save the model parameters.
        if (_jsonParameters != null) {
            oos.writeObject(_jsonParameters.toString());
        } else {
            oos.writeObject(null);
        }
        // Save the _switchProbabilities part.
        oos.writeObject(_switchProbabilities);

        // Save the model if it has been trained.
        if (_model != null) {
            // Create a (likely) unique file name for the liblinear model.
            String fileName =
                Long.toHexString(Double.doubleToLongBits(Math.random())) +
                ".jliblinear";

            File file = new File(fileName);
            // Save the liblinear model to a separate file.
            Linear.saveModel(file, _model);
            // Save the model file name.
            oos.writeObject(fileName);
        } else {
            // The model has not been trained.
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        // Load the model parameters.
        String jsonText = (String)ois.readObject();
        if (jsonText != null) {
            _jsonParameters = new JSONObject(jsonText);
        }
        // Load the _switchProbabilities part.
        _switchProbabilities = (boolean)ois.readObject();

        // Load model file name from the Java input stream.
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
