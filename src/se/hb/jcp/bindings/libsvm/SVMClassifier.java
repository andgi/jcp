// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2015  Anders Gidenstam
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
package se.hb.jcp.bindings.libsvm;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.IClassifier;
import se.hb.jcp.ml.IClassProbabilityClassifier;

public class SVMClassifier
    implements IClassProbabilityClassifier,
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
                              ("se.hb.jcp.bindings.libsvm.SVMClassifier: " +
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
                              ("se.hb.jcp.bindings.libsvm.SVMClassifier: " +
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
            _parameters.gamma = 1.0/2; // FIXME: Should be 1/#classes
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
        if (x instanceof se.hb.jcp.bindings.libsvm.SparseDoubleMatrix2D) {
            tmp_x = (SparseDoubleMatrix2D)x;
        } else {
            tmp_x = new SparseDoubleMatrix2D(x.rows(), x.columns());
            tmp_x.assign(x);
        }

        _model = svm.svm_train(_parameters, tmp_x, y);
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
        if (instance instanceof
                se.hb.jcp.bindings.libsvm.SparseDoubleMatrix1D) {
            tmp_instance = (SparseDoubleMatrix1D)instance;
        } else {
            tmp_instance = new SparseDoubleMatrix1D(instance.size());
            tmp_instance.assign(instance);
        }

        return svm.svm_predict(_model, tmp_instance);
    }

    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates)
    {
        SparseDoubleMatrix1D tmp_instance;
        if (instance instanceof
                se.hb.jcp.bindings.libsvm.SparseDoubleMatrix1D) {
            tmp_instance = (SparseDoubleMatrix1D)instance;
        } else {
            tmp_instance = new SparseDoubleMatrix1D(instance.size());
            tmp_instance.assign(instance);
        }

        double prediction = svm.svm_predict_probability(_model,
                                                        tmp_instance,
                                                        probabilityEstimates);
        // libsvm seems to use the opposite order of labels, so reverse
        // the array of probability estimates before returning them.
        // FIXME: Verify for more data sets.
        int i = 0;
        int j = probabilityEstimates.length-1;
        for (; i < j; i++, j--) {
            double tmp = probabilityEstimates[i];
            probabilityEstimates[i] = probabilityEstimates[j];
            probabilityEstimates[j] = tmp;
        }

        return prediction;
    }

    public int getAttributeCount()
    {
        return _attributeCount;
    }

    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }
}
