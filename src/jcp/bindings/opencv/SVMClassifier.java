// Copyright (C) 2014  Anders Gidenstam
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
    protected CvSVMParams _parameters;

    public SVMClassifier()
    {
        this((CvSVMParams)null);
    }

    public SVMClassifier(JSONObject parameters)
    {
        this();
        if (parameters.has("svm_type")) {
            String type = parameters.getString("svm_type");
            if (type.equals("C_SVC")) {
                _parameters.set_svm_type(CvSVM.C_SVC);
            } else if (type.equals("NU_SVC")) {
                _parameters.set_svm_type(CvSVM.NU_SVC);
            } else if (type.equals("ONE_CLASS")) {
                _parameters.set_svm_type(CvSVM.ONE_CLASS);
            } else if (type.equals("EPSILON_SVR")) {
                _parameters.set_svm_type(CvSVM.EPS_SVR);
            } else if (type.equals("NU_SVR")) {
                _parameters.set_svm_type(CvSVM.NU_SVR);
            } else {
                throw new IllegalArgumentException
                              ("jcp.bindings.opencv.SVMClassifier: " +
                               "Unknown svm_type '" + type + "'.");
            }
        }
        if (parameters.has("kernel_type")) {
            String type = parameters.getString("kernel_type");
            if (type.equals("LINEAR")) {
                _parameters.set_kernel_type(CvSVM.LINEAR);
            } else if (type.equals("POLY")) {
                _parameters.set_kernel_type(CvSVM.POLY);
            } else if (type.equals("RBF")) {
                _parameters.set_kernel_type(CvSVM.RBF);
            } else if (type.equals("SIGMOID")) {
                _parameters.set_kernel_type(CvSVM.SIGMOID);
            } else {
                throw new IllegalArgumentException
                              ("jcp.bindings.opencv.SVMClassifier: " +
                               "Unknown kernel_type '" + type + "'.");
            }
        }
        if (parameters.has("degree")) {
            _parameters.set_degree(parameters.getDouble("degree"));
        }
        if (parameters.has("gamma")) {
            _parameters.set_gamma(parameters.getDouble("gamma"));
        }
        if (parameters.has("coef0")) {
            _parameters.set_coef0(parameters.getDouble("coef0"));
        }
        if (parameters.has("C")) {
            _parameters.set_C(parameters.getDouble("C"));
        }
        if (parameters.has("nu")) {
            _parameters.set_nu(parameters.getDouble("nu"));
        }
        if (parameters.has("p")) {
            _parameters.set_p(parameters.getDouble("p"));
        }
        if (parameters.has("termination_criteria")) {
            JSONObject termination =
                parameters.getJSONObject("termination_criteria");
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
            _parameters.set_term_crit(new TermCriteria(criteria,
                                                       max_iter,
                                                       epsilon));
        }
    }

    public SVMClassifier(CvSVMParams parameters)
    {
        _model = new CvSVM();
        _parameters = parameters;
        if (_parameters == null) {
            // Default SVM parameters.
            _parameters = new CvSVMParams();
            _parameters.set_svm_type(CvSVM.C_SVC);
            _parameters.set_kernel_type(CvSVM.RBF);
            _parameters.set_degree(3);
            _parameters.set_gamma(1.0/2); // FIXME: Should be 1/#classes
            _parameters.set_coef0(0);
            _parameters.set_nu(0.5);
            _parameters.set_C(1);
            _parameters.set_term_crit(new TermCriteria(TermCriteria.EPS,
                                                       0,
                                                       1e-3));
            _parameters.set_p(0.1);
        }
    }

    public void fit(DoubleMatrix2D x, double[] y)
    {
        ((CvSVM)_model).train(asDDM2D(x).asMat(),
                              asDDM1D(y).asMat(),
                              new org.opencv.core.Mat(),
                              new org.opencv.core.Mat(),
                              _parameters);
        _attributeCount = x.columns();
        // FIXME: Fit and forget. Ugly hack due to CvSVMParams not being
        //        serializable.
        _parameters = null;
    }

    public IClassifier fitNew(DoubleMatrix2D x, double[] y)
    {
        // FIXME: If fit() has been called on the parent model the
        //        _parameters attribute has been reset to null.
        SVMClassifier clone = new SVMClassifier(_parameters);
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
}
