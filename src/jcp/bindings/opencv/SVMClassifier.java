// Copyright (C) 2014 - 2015  Anders Gidenstam
// License: to be defined.
package jcp.bindings.opencv;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.opencv.ml.CvStatModel;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.core.TermCriteria;

import org.json.JSONObject;

import jcp.ml.IClassifier;

public class SVMClassifier
    extends ClassifierBase
    implements java.io.Serializable
{
    public SVMClassifier()
    {
    }

    public SVMClassifier(JSONObject parameters)
    {
        this();
        _jsonParameters = parameters;
    }

    public void fit(DoubleMatrix2D x, double[] y)
    {
        CvSVMParams parameters = readParameters();
        if (_model == null) {
            _model = new CvSVM();
        }
        ((CvSVM)_model).train(asDDM2D(x).asMat(),
                              asDDM1D(y).asMat(),
                              new org.opencv.core.Mat(),
                              new org.opencv.core.Mat(),
                              parameters);
        _attributeCount = x.columns();
    }

    public IClassifier fitNew(DoubleMatrix2D x, double[] y)
    {
        SVMClassifier clone = new SVMClassifier(_jsonParameters);
        clone.fit(x, y);
        return clone;
    }

    public double predict(DoubleMatrix1D instance)
    {
        return ((CvSVM)_model).predict(asDDM1D(instance).asMat());
    }

    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates)
    {
        // FIXME: This method is not properly implemented yet. It might work
        //        for {-1, 1} two-class problems.
        double prediction = ((CvSVM)_model).predict(asDDM1D(instance).asMat());
        probabilityEstimates[0] = 0.5 + 0.5*prediction;
        probabilityEstimates[1] = 0.5 - 0.5*prediction;
        return prediction;
    }

    protected CvStatModel getNewInstance()
    {
        return new CvSVM();
    }

    private CvSVMParams readParameters()
    {
        CvSVMParams parameters = new CvSVMParams();

        // Default SVM parameters.
        parameters = new CvSVMParams();
        parameters.set_svm_type(CvSVM.C_SVC);
        parameters.set_kernel_type(CvSVM.RBF);
        parameters.set_degree(3);
        parameters.set_gamma(1.0/2); // FIXME: Should be 1/#classes
        parameters.set_coef0(0);
        parameters.set_nu(0.5);
        parameters.set_C(1);
        parameters.set_term_crit(new TermCriteria(TermCriteria.EPS,
                                                  0,
                                                  1e-3));
        parameters.set_p(0.1);

        if (_jsonParameters != null) {
            if (_jsonParameters.has("svm_type")) {
                String type = _jsonParameters.getString("svm_type");
                if (type.equals("C_SVC")) {
                    parameters.set_svm_type(CvSVM.C_SVC);
                } else if (type.equals("NU_SVC")) {
                    parameters.set_svm_type(CvSVM.NU_SVC);
                } else if (type.equals("ONE_CLASS")) {
                    parameters.set_svm_type(CvSVM.ONE_CLASS);
                } else if (type.equals("EPSILON_SVR")) {
                    parameters.set_svm_type(CvSVM.EPS_SVR);
                } else if (type.equals("NU_SVR")) {
                    parameters.set_svm_type(CvSVM.NU_SVR);
                } else {
                    throw new IllegalArgumentException
                                  ("jcp.bindings.opencv.SVMClassifier: " +
                                   "Unknown svm_type '" + type + "'.");
                }
            }
            if (_jsonParameters.has("kernel_type")) {
                String type = _jsonParameters.getString("kernel_type");
                if (type.equals("LINEAR")) {
                    parameters.set_kernel_type(CvSVM.LINEAR);
                } else if (type.equals("POLY")) {
                    parameters.set_kernel_type(CvSVM.POLY);
                } else if (type.equals("RBF")) {
                    parameters.set_kernel_type(CvSVM.RBF);
                } else if (type.equals("SIGMOID")) {
                    parameters.set_kernel_type(CvSVM.SIGMOID);
                } else {
                    throw new IllegalArgumentException
                                  ("jcp.bindings.opencv.SVMClassifier: " +
                                   "Unknown kernel_type '" + type + "'.");
                }
            }
            if (_jsonParameters.has("degree")) {
                parameters.set_degree(_jsonParameters.getDouble("degree"));
            }
            if (_jsonParameters.has("gamma")) {
                parameters.set_gamma(_jsonParameters.getDouble("gamma"));
            }
            if (_jsonParameters.has("coef0")) {
                parameters.set_coef0(_jsonParameters.getDouble("coef0"));
            }
            if (_jsonParameters.has("C")) {
                parameters.set_C(_jsonParameters.getDouble("C"));
            }
            if (_jsonParameters.has("nu")) {
                parameters.set_nu(_jsonParameters.getDouble("nu"));
            }
            if (_jsonParameters.has("p")) {
                parameters.set_p(_jsonParameters.getDouble("p"));
            }
            if (_jsonParameters.has("termination_criteria")) {
                JSONObject termination =
                    _jsonParameters.getJSONObject("termination_criteria");
                int criteria = 0;
                int max_iter = 0;
                double epsilon = 0.0;
                if (termination.has("max_count")) {
                    criteria += TermCriteria.MAX_ITER;
                    max_iter = termination.getInt("max_iter");
                }
                if (termination.has("epsilon")) {
                    criteria += TermCriteria.EPS;
                    epsilon = termination.getDouble("epsilon");
                }
                parameters.set_term_crit(new TermCriteria(criteria,
                                                          max_iter,
                                                          epsilon));
            }
        }
        return parameters;
    }

}
