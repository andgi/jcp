// Copyright (C) 2014 - 2015  Anders Gidenstam
// License: to be defined.
package jcp.bindings.opencv;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvStatModel;
import org.opencv.ml.CvRTrees;
import org.opencv.ml.CvRTParams;

import org.json.JSONObject;

import jcp.ml.IClassifier;

public class RandomForestClassifier
    extends ClassifierBase
    implements java.io.Serializable
{
    public RandomForestClassifier()
    {
    }

    public RandomForestClassifier(JSONObject parameters)
    {
        this();
        _jsonParameters = parameters;
    }

    public void fit(DoubleMatrix2D x, double[] y)
    {
        CvRTParams parameters = readParameters();
        if (_model == null) {
            _model = new CvRTrees();
        }
        // Setup the variable and sample indexes etc to use all data.
        Mat varIdx = new MatOfInt();
        Mat sampleIdx = new MatOfInt();
        Mat varType = new MatOfInt();
        Mat missingDataMask = new MatOfInt();

        ((CvRTrees)_model).train(asDDM2D(x).asMat(),
                                 1, // should be CV_ROW_SAMPLE enum/constant
                                 asDDM1D(y).asMat(),
                                 varIdx,
                                 sampleIdx,
                                 varType,
                                 missingDataMask,
                                 parameters);
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

    private CvRTParams readParameters()
    {
        CvRTParams parameters = new CvRTParams();

        // Set default RandomForest parameters.
        // FIXME?

        // RandomForest parameters based on CvRTParams.
        if (_jsonParameters != null) {
            if (_jsonParameters.has("max_depth")) {
                parameters.set_max_depth(_jsonParameters.getInt("max_depth"));
            }
            if (_jsonParameters.has("min_sample_count")) {
                parameters.set_min_sample_count
                    (_jsonParameters.getInt("min_sample_count"));
            }
            if (_jsonParameters.has("use_surrogates")) {
                // FIXME: Accept boolean text input?
                parameters.set_use_surrogates
                    (_jsonParameters.getInt("use_surrogates") != 0);
            }
            if (_jsonParameters.has("max_categories")) {
                parameters.set_max_categories
                    (_jsonParameters.getInt("max_categories"));
            }
            if (_jsonParameters.has("calc_var_importance")) {
                parameters.set_calc_var_importance
                    (_jsonParameters.getInt("calc_var_importance") != 0);
            }
            if (_jsonParameters.has("nactive_vars")) {
                parameters.set_nactive_vars
                    (_jsonParameters.getInt("nactive_vars"));
            }
            // FIXME: Cannot be set?
            if (_jsonParameters.has("max_num_of_trees")) {
                //parameters.set_max_num_of_trees_in_the_forest
                //    (_jsonParameters.getInt("max_num_of_trees"));
            }
            // FIXME: Cannot be set?
            if (_jsonParameters.has("forest_accuracy")) {
                //parameters.set_forest_accuracy
                //    (_jsonParameters.getInt("forest_accuracy"));
            }
            if (_jsonParameters.has("priors")) {
                // FIXME: Implement?
            }
            TermCriteria criteria = readTerminationCriteria();
            if (criteria != null) {
                parameters.set_term_crit(criteria);
            }
        }
        return parameters;
    }
}
