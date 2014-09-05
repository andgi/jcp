// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.opencv;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

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
    }

    public void fit(DoubleMatrix2D x, double[] y)
    {
        ((CvSVM)_model).train(asDDM2D(x).asMat(),
                              asDDM1D(y).asMat());
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
}
