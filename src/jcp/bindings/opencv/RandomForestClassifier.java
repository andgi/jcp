// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.opencv;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.opencv.ml.CvStatModel;
import org.opencv.ml.CvRTrees;
import org.opencv.ml.CvRTParams;

import jcp.ml.IClassifier;

public class RandomForestClassifier
    extends ClassifierBase
    implements java.io.Serializable
{
    protected CvRTParams _parameters;

    public RandomForestClassifier()
    {
        _model = new CvRTrees();
        _parameters = null;
    }

    public void fit(DoubleMatrix2D x, double[] y)
    {
        ((CvRTrees)_model).train(asDDM2D(x).asMat(),
                                 1, // should be CV_ROW_SAMPLE enum/constant
                                 asDDM1D(y).asMat());
        _attributeCount = x.columns();
    }

    public IClassifier fitNew(DoubleMatrix2D x, double[] y)
    {
        RandomForestClassifier clone = new RandomForestClassifier();
        clone.fit(x, y);
        return clone;
    }

    public double predict(DoubleMatrix1D instance)
    {
        return ((CvRTrees)_model).predict(asDDM1D(instance).asMat());
    }

    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates)
    {
        // FIXME: This method is not properly implemented yet. It might work
        //        for {-1, 1} two-class problems.
        double prediction =
            ((CvRTrees)_model).predict(asDDM1D(instance).asMat());
        probabilityEstimates[0] = 0.5 + 0.5*prediction;
        probabilityEstimates[1] = 0.5 - 0.5*prediction;
        return prediction;
    }

    protected CvStatModel getNewInstance()
    {
        return new CvRTrees();
    }
}
