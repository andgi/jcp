// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.opencv;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.opencv.ml.CvStatModel;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.core.TermCriteria;

import jcp.ml.IClassifier;

public class SVMClassifier
    extends ClassifierBase
    implements java.io.Serializable
{
    protected CvSVMParams _parameters;

    public SVMClassifier()
    {
        this(null);
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
        // FIXME: Fit and forget. Ugly hack due to CvSVMParams not being
        //        serializable.
        _parameters = null;
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
