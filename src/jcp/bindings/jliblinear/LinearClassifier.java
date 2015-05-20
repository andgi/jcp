// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.bindings.jliblinear;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.File;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import de.bwaldvogel.liblinear.*;

import jcp.ml.IClassifier;

public class LinearClassifier
    implements IClassifier,
               java.io.Serializable
{
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected Parameter _parameters;
    protected Model _model;
    protected int _attributeCount = -1;

    public LinearClassifier()
    {
        this((Parameter)null);
    }

    public LinearClassifier(JSONObject parameters)
    {
        this();
        if (parameters.has("solver_type")) {
            String type = parameters.getString("solver_type");
            if (type.equals("L2R_LR")) {
                _parameters.setSolverType(SolverType.L2R_LR);
            } else if (type.equals("L2R_L2LOSS_SVC_DUAL")) {
                _parameters.setSolverType(SolverType.L2R_L2LOSS_SVC_DUAL);
            } else if (type.equals("L2R_L2LOSS_SVC")) {
                _parameters.setSolverType(SolverType.L2R_L2LOSS_SVC);
            } else if (type.equals("L2R_L1LOSS_SVC_DUAL")) {
                _parameters.setSolverType(SolverType.L2R_L1LOSS_SVC_DUAL);
            } else if (type.equals("MCSVM_CS")) {
                _parameters.setSolverType(SolverType.MCSVM_CS);
            } else if (type.equals("L1R_L2LOSS_SVC")) {
                _parameters.setSolverType(SolverType.L1R_L2LOSS_SVC);
            } else if (type.equals("L1R_LR")) {
                _parameters.setSolverType(SolverType.L1R_LR);
            } else if (type.equals("L2R_LR_DUAL")) {
                _parameters.setSolverType(SolverType.L2R_LR_DUAL);
            } else if (type.equals("L2R_L2LOSS_SVR")) {
                _parameters.setSolverType(SolverType.L2R_L2LOSS_SVR);
            } else if (type.equals("L2R_L2LOSS_SVR_DUAL")) {
                _parameters.setSolverType(SolverType.L2R_L2LOSS_SVR_DUAL);
            } else if (type.equals("L2R_L1LOSS_SVR_DUAL")) {
                _parameters.setSolverType(SolverType.L2R_L1LOSS_SVR_DUAL);
            } else {
                throw new IllegalArgumentException
                              ("jcp.bindings.jliblinear.LinearClassifier: " +
                               "Unknown SolverType '" + type + "'.");
            }
        }
        if (parameters.has("C")) {
            _parameters.setC(parameters.getDouble("C"));
        }
        if (parameters.has("eps")) {
            _parameters.setEps(parameters.getDouble("eps"));
        }
        if (parameters.has("p")) {
            _parameters.setP(parameters.getDouble("p"));
        }
    }

    public LinearClassifier(Parameter parameters)
    {
        _parameters = parameters;

        if (_parameters == null) {
            // Default parameters.
            _parameters = new Parameter(SolverType.L2R_LR,
                                        1.0,
                                        0.01);
        }
    }

    public void fit(DoubleMatrix2D x, double[] y)
    {
        SparseDoubleMatrix2D tmp_x;
        if (x instanceof SparseDoubleMatrix2D) {
            tmp_x = (SparseDoubleMatrix2D)x;
        } else {
            tmp_x = new SparseDoubleMatrix2D(x.rows(), x.columns());
            tmp_x.assign(x);
        }
        Problem problem = new Problem();
        problem.bias = 0.0;
        problem.l = y.length;
        problem.n = tmp_x.columns();
        problem.x = tmp_x.rows;
        problem.y = y;

        _model = Linear.train(problem, _parameters);
        _attributeCount = tmp_x.columns();
    }

    public IClassifier fitNew(DoubleMatrix2D x, double[] y)
    {
        LinearClassifier clone = new LinearClassifier(_parameters);
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

        System.err.println("Linear.predictProbability() ... " + _model);

        return Linear.predict(_model, tmp_instance.nodes);
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

        return Linear.predictProbability(_model,
                                         tmp_instance.nodes,
                                         probabilityEstimates);
    }

    public int getAttributeCount()
    {
        return _attributeCount;
    }

    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        // Create a (likely) unique file name for the liblinear model.
        String fileName =
            Long.toHexString(Double.doubleToLongBits(Math.random())) +
            ".jliblinear";

        File file = new File(fileName);
        // Write the liblinear model to a separate file.
        Linear.saveModel(file, _model);

        // Write the attribute count and model file name to the
        // Java output stream.
        oos.writeObject(_attributeCount);
        oos.writeObject(fileName);
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        // Load the attribute count.
        _attributeCount = (int)ois.readObject();
        // Load model file name from the Java input stream.
        String fileName = (String)ois.readObject();

        // Load the liblinear model from the designated file.
        File file = new File(fileName);
        // Load the model from a separate file.
        _model = Linear.loadModel(file);
    }
}
