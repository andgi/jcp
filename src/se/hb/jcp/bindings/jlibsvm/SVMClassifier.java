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
package se.hb.jcp.bindings.jlibsvm;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.concurrent.atomic.AtomicReference;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import libsvm.*;

import se.hb.jcp.ml.IClassifier;
import se.hb.jcp.ml.ISVMClassifier;
import se.hb.jcp.ml.IClassProbabilityClassifier;
import se.hb.jcp.ml.ClassifierBase;

public class SVMClassifier
    extends ClassifierBase
    implements ISVMClassifier, //IClassProbabilityClassifier // FIXME: disabled.
               java.io.Serializable
{
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected svm_parameter _parameters;
    protected svm_model _model;
    AtomicReference<double[]> _cachedW = new AtomicReference<double[]>();

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

    protected void internalFit(DoubleMatrix2D x, double[] y)
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

        double prediction = svm.svm_predict_probability(_model,
                                                        tmp_instance.nodes,
                                                        probabilityEstimates);
        // jlibsvm seem to use the reverse order of labels, so reverse
        // the array of probability estimates before returning them.
        // FIXME: Verify for more data sets. Use svm_model.label[c] and
        //        this.getLabels() to ensure compatibility.
        // FIXME: Disabled for the time being as it seems jlibsvm
        //        always return the same probabilities for all instances.
        //        An issue in the default configuration?
        int i = 0;
        int j = probabilityEstimates.length-1;
        for (; i < j; i++, j--) {
            double tmp = probabilityEstimates[i];
            probabilityEstimates[i] = probabilityEstimates[j];
            probabilityEstimates[j] = tmp;
        }
        return prediction;
    }

    /**
     * Returns the signed distance between the separating hyperplane and the
     * instance.
     *
     * @return the signed distance between the separating hyperplane and the instance.
     */
    public double distanceFromSeparatingPlane(DoubleMatrix1D instance)
    {
        // FIXME: This is only valid for 2-class SVM and classes -1.0 and 1.0.
        double[] w = getW();
        double   b = computeB();
        if (_model.nr_class == 2) {
            double distance = b;
            for (int i = 0; i < instance.size(); i++) {
                distance += w[i] * instance.get(i);
            }
            return distance;
        } else {
            throw new UnsupportedOperationException("Not implemented");
        }
    }

    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }

    /**
     * Returns the signed distance from the origin to the separating
     * hyperplane. See the libSVM FAQ #804,
     * http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f804 for the internal
     * details.
     *
     * @return the signed distance from the origin to the separating hyperplane.
     */
    private double computeB()
    {
        // FIXME: This is only valid for 2-class SVM and classes -1.0 and 1.0.
        double sign = _model.label[0] == -1.0 ? -1.0 : 1.0;
        return sign * -_model.rho[0];
    }

    private double[] getW()
    {
        double[] w = _cachedW.get();
        if (w == null) {
            w = computeW();
            _cachedW.compareAndSet(null, w);
        }
        return w;
    }

    /**
     * Returns a normal vector of the separating hyperplane.
     * See the libSVM FAQ #804,
     * http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f804 for the internal
     * details.
     *
     * @return a normal vector of the separating hyperplane.
     */
    private double[] computeW()
    {
        // FIXME: This is only valid for 2-class SVM and classes -1.0 and 1.0.
        double sign = _model.label[0] == -1.0 ? -1.0 : 1.0;
        double[] w = new double[getAttributeCount()];
        for (int l = 0; l < _model.l; l++) {
            for (int i = 0; i < _model.SV[l].length; i++) {
                w[_model.SV[l][i].index] +=
                    sign * _model.sv_coef[0][l] * _model.SV[l][i].value;
            }
        }
        return w;
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        // Save the classifier parameters.
        oos.writeObject(_parameters);
        oos.writeObject(_cachedW);
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

    @SuppressWarnings("unchecked")
    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        // Load the classifier parameters.
        _parameters = (svm_parameter)ois.readObject();
        _cachedW = (AtomicReference<double[]>)ois.readObject();
        // Load the model file name from the Java input stream.
        String fileName = (String)ois.readObject();
        if (fileName != null) {
            // Load the model from the designated file.
            _model = svm.svm_load_model(fileName);
        }
    }

    static {
        // Replace the svm print_string_function with a no-op.
        svm.svm_set_print_string_function(new libsvm.svm_print_interface() {
            public void print(String s) {
            }
        });
    }
}
