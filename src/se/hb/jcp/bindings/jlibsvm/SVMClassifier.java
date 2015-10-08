// JCP - Java Conformal Prediction framework
// Copyright (C) 2015  Anders Gidenstam
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
package se.hb.jcp.bindings.jlibsvm;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import libsvm.*;

import se.hb.jcp.ml.IClassifier;

public class SVMClassifier
    implements IClassifier,
               java.io.Serializable
{
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected svm_parameter _parameters;
    protected svm_model _model;
    protected int _attributeCount = -1;

    public SVMClassifier()
    {
        this((svm_parameter)null);
    }

    public SVMClassifier(JSONObject parameters)
    {
        this();
        if (parameters.has("svm_type")) {
            String type = parameters.getString("svm_type");
            if (type.equals("C_SVC")) {
                _parameters.svm_type = svm_parameter.C_SVC;
            } else if (type.equals("NU_SVC")) {
                _parameters.svm_type = svm_parameter.NU_SVC;
            } else if (type.equals("ONE_CLASS")) {
                _parameters.svm_type = svm_parameter.ONE_CLASS;
            } else if (type.equals("EPSILON_SVR")) {
                _parameters.svm_type = svm_parameter.EPSILON_SVR;
            } else if (type.equals("NU_SVR")) {
                _parameters.svm_type = svm_parameter.NU_SVR;
            } else {
                throw new IllegalArgumentException
                              ("se.hb.jcp.bindings.jlibsvm.SVMClassifier: " +
                               "Unknown svm_type '" + type + "'.");
            }
        }
        if (parameters.has("kernel_type")) {
            String type = parameters.getString("kernel_type");
            if (type.equals("LINEAR")) {
                _parameters.kernel_type = svm_parameter.LINEAR;
            } else if (type.equals("POLY")) {
                _parameters.kernel_type = svm_parameter.POLY;
            } else if (type.equals("RBF")) {
                _parameters.kernel_type = svm_parameter.RBF;
            } else if (type.equals("SIGMOID")) {
                _parameters.kernel_type = svm_parameter.SIGMOID;
            } else if (type.equals("PRECOMPUTED")) {
                _parameters.kernel_type = svm_parameter.PRECOMPUTED;
            } else {
                throw new IllegalArgumentException
                              ("se.hb.jcp.bindings.jlibsvm.SVMClassifier: " +
                               "Unknown kernel_type '" + type + "'.");
            }
        }
        if (parameters.has("degree")) {
            _parameters.degree = (int)parameters.getDouble("degree");
        }
        if (parameters.has("gamma")) {
            _parameters.gamma = parameters.getDouble("gamma");
        }
        if (parameters.has("coef0")) {
            _parameters.coef0 = parameters.getDouble("coef0");
        }
        if (parameters.has("C")) {
            _parameters.C = parameters.getDouble("C");
        }
        if (parameters.has("nu")) {
            _parameters.nu = parameters.getDouble("nu");
        }
        if (parameters.has("p")) {
            _parameters.p = parameters.getDouble("p");
        }
        if (parameters.has("cache_size")) {
            _parameters.cache_size = parameters.getDouble("cache_size");
        }
        if (parameters.has("shrinking")) {
            _parameters.shrinking = parameters.getInt("shrinking");
        }
        if (parameters.has("estimate_probability")) {
            _parameters.probability = parameters.getInt("estimate_probability");
        }
        if (parameters.has("termination_criteria")) {
            JSONObject termination =
                parameters.getJSONObject("termination_criteria");
            if (termination.has("epsilon")) {
                _parameters.eps = termination.getDouble("epsilon");
            }
        }
    }

    public SVMClassifier(svm_parameter parameters)
    {
        _parameters = parameters;

        if (_parameters == null) {
            // Default libsvm parameter.
            _parameters = new svm_parameter();
            _parameters.svm_type = svm_parameter.C_SVC;
            _parameters.kernel_type = svm_parameter.RBF;
            _parameters.degree = 3;
            _parameters.gamma = 1.0/500; // FIXME: Should be 1/#attributes
            _parameters.coef0 = 0;
            _parameters.nu = 0.5;
            _parameters.cache_size = 100;
            _parameters.C = 1;
            _parameters.eps = 1e-3;
            _parameters.p = 0.1;
            _parameters.shrinking = 1;
            _parameters.probability = 1; // Must be set for the NC-function.
            _parameters.nr_weight = 0;
            _parameters.weight_label = new int[0];
            _parameters.weight = new double[0];
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
        svm_problem problem = new svm_problem();
        problem.l = y.length;
        problem.x = tmp_x.rows;
        problem.y = y;

        _model = svm.svm_train(problem, _parameters);
        _attributeCount = tmp_x.columns();
    }

    public IClassifier fitNew(DoubleMatrix2D x, double[] y)
    {
        SVMClassifier clone = new SVMClassifier(_parameters);
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

        return svm.svm_predict(_model, tmp_instance.nodes);
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

        return svm.svm_predict_probability(_model,
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
        // Save the classifier parameters.
        oos.writeObject(_parameters);
        // Save the attribute count.
        oos.writeObject(_attributeCount);
        // Save the model if it has been trained.
        if (_model != null) {
            // Create a (likely) unique file name for the Java libsvm model.
            String fileName =
                Long.toHexString(Double.doubleToLongBits(Math.random())) +
                ".jlibsvm";

            // Save the model to a separate file.
            svm.svm_save_model(fileName, _model);
            // Save the model file name.
            oos.writeObject(fileName);
        } else {
            // Save null if the model has not been trained.
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        // Load the classifier parameters.
        _parameters = (svm_parameter)ois.readObject();
        // Load the attribute count.
        _attributeCount = (int)ois.readObject();
        // Load the model file name from the Java input stream.
        String fileName = (String)ois.readObject();
        if (fileName != null) {
            // Load the model from the designated file.
            _model = svm.svm_load_model(fileName);
        }
    }
}
