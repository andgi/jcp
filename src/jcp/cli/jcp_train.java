// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.cli;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Date;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.Random;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.ObjectMatrix2D;

import jcp.cp.*;
import jcp.nc.*;
import jcp.io.*;

/**
 * Command line training tool for JCP.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class jcp_train
{
    private DataSet _full;
    private SortedSet<Double> _classSet;
    private double[]          _classes;
    private DataSet _training;
    private DataSet _calibration;
    private DataSet _test;

    private final double SIGNIFICANCE_LEVEL = 0.10;

    public jcp_train()
    {
        super();
        _training    = new DataSet();
        _calibration = new DataSet();
        _test        = new DataSet();
    }

    public void run(String[] args)
        throws IOException
    {
        // Load and create training and calibration sets.
        if (args.length != 1) {
            System.out.println("Usage: jcp_train <libsvm formatted data set>");
            System.exit(-1);
        }
        // FIXME: Only initial use case yet. Train & test.
        runICCTest(args);
        //runPlainSVMTest(args);
    }

    private void runICCTest(String[] args)
        throws IOException
    {
        long t1 = System.currentTimeMillis();
        loadDataset(args[0]);
        extractClasses();
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");

        splitDataset(0.4, 0.1);
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        System.out.println("Training on " + _training.x.rows() +
                           " instances and calibrating on " +
                           _calibration.x.rows() +
                           " instances.");


        InductiveConformalClassifier icc =
            new InductiveConformalClassifier(_classes);
        icc._nc =
            new SVMClassificationNonconformityFunction(_classes);

        icc.fit(_training.x, _training.y, _calibration.x, _calibration.y);
        long t4 = System.currentTimeMillis();
        System.out.println("Training complete.");
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");

        System.out.println("Testing accuracy on " + _test.x.rows() +
                           " instances at a significance level of " +
                           SIGNIFICANCE_LEVEL + ".");

        // Evaluation on the test set.
        ObjectMatrix2D pred = null;
        pred = icc.predict(_test.x, SIGNIFICANCE_LEVEL);
        //System.out.println(pred);

        int correct = 0;
        int[] correctAtSize = new int[_classSet.size()+1];
        int[] predictionAtSize = new int[_classSet.size()+1];

        for (int i = 0; i < pred.rows(); i++){
            int classIndex = _classSet.headSet(_test.y[i]).size();
            int predictionSize = 0;
            for (int c = 0; c < _classes.length; c++) {
                if ((Boolean)pred.get(i, c)) {
                    predictionSize++;
                }
            }
            predictionAtSize[predictionSize]++;

            if ((Boolean)pred.get(i, classIndex)) {
                correct++;
                correctAtSize[predictionSize]++;
            }
        }
        long t5 = System.currentTimeMillis();

        System.out.println("Accuracy " + ((double)correct / _test.y.length));
        for (int s = 0; s < predictionAtSize.length; s++) {
            System.out.println("  #Predictions with " + s + " classes: " +
                               predictionAtSize[s] + ". Accuracy: " +
                               (double)correctAtSize[s]/(double)predictionAtSize[s]);
        }
        System.out.println("Duration " + (double)(t5 - t4)/1000.0 + " sec.");
        System.out.println("Total Duration " + (double)(t5 - t1)/1000.0 +
                           " sec.");
    }

    private void runPlainSVMTest(String[] args)
        throws IOException
    {
        long t1 = System.currentTimeMillis();
        loadDataset(args[0]);
        extractClasses();
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");

        splitDataset(0.4, 0.0);
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        System.out.println("Training on " + _training.x.rows() +
                           " instances and calibrating on " +
                           _calibration.x.rows() +
                           " instances.");
        // Default libsvm parameter.
        jcp.bindings.libsvm.svm_parameter parameter =
            new jcp.bindings.libsvm.svm_parameter();
        parameter.svm_type = jcp.bindings.libsvm.svm_parameter.C_SVC;
        parameter.kernel_type = jcp.bindings.libsvm.svm_parameter.RBF;
        parameter.degree = 3;
        parameter.gamma = 1.0/_classes.length;
        parameter.coef0 = 0;
        parameter.nu = 0.5;
        parameter.cache_size = 100;
        parameter.C = 1;
        parameter.eps = 1e-3;
        parameter.p = 0.1;
        parameter.shrinking = 1;
        parameter.probability = 1;
        parameter.nr_weight = 0;
        parameter.weight_label = new int[0];
        parameter.weight = new double[0];

        jcp.bindings.libsvm.svm_model model =
            jcp.bindings.libsvm.svm.svm_train
                (parameter,
                 (jcp.bindings.libsvm.SparseDoubleMatrix2D)_training.x,
                 _training.y);
        long t4 = System.currentTimeMillis();
        System.out.println("Training complete.");
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");

        System.out.println("Testing accuracy on " + _test.x.rows() +
                           " instances.");

        // Evaluation on the test set.
        int correct = 0;
        double[] probability = new double[_classes.length];

        for (int i = 0; i < _test.x.rows(); i++) {
            int classIndex = _classSet.headSet(_test.y[i]).size();
            double prediction =
                jcp.bindings.libsvm.svm.svm_predict_probability
                    (model,
                     (jcp.bindings.libsvm.SparseDoubleMatrix1D)
                         _test.x.viewRow(i),
                     probability);
            if (prediction == _test.y[i]) {
                correct++;
            }
        }
        long t5 = System.currentTimeMillis();

        System.out.println("Accuracy " + ((double)correct / _test.y.length));
        System.out.println("Duration " + (double)(t5 - t4)/1000.0 + " sec.");
        System.out.println("Total Duration " + (double)(t5 - t1)/1000.0 +
                           " sec.");
    }

    private void loadDataset(String filename)
        throws IOException
    {
        FileInputStream file;
        file = new FileInputStream(filename);

        _full =
            new libsvmReader().read
                (file,
                 // Select the desired representation.
                 new cern.colt.matrix.impl.SparseDoubleMatrix2D(0, 0));
                 //new jcp.bindings.libsvm.SparseDoubleMatrix2D(0, 0));
        file.close();

        System.out.println("Loaded the dataset " + filename + " containing " +
                           _full.x.rows() + " instances with " +
                           _full.x.columns() + " attributes.");
    }

    private void extractClasses()
    {
        _classSet = new TreeSet<Double>();
        for (int r = 0; r < _full.x.rows(); r++) {
            if (!_classSet.contains(_full.y[r])) {
                _classSet.add(_full.y[r]);
            }
        }
        _classes = new double[_classSet.size()];
        System.out.println("Classes: ");
        int i = 0;
        for (Double c : _classSet.toArray(new Double[0])) {
            _classes[i] = c;
            System.out.println("   " + _classes[i]);
            i++;
        }
    }

    private void splitDataset(double trainingFraction,
                              double calibrationFraction)
    {
        // Set template matrix types for the split data set.
        _training.x = new jcp.bindings.libsvm.SparseDoubleMatrix2D(0, 0);
        _calibration.x = _training.x;
        _test.x = _training.x;
        // Partition the full data set into training, calibartion and test sets.
        _full.random3Partition(_training, _calibration, _test,
                               trainingFraction, calibrationFraction);
        System.out.println("Split the data set into training set, " +
                           _training.x.rows() + " instances, " +
                           "calibration set, " +
                           _calibration.x.rows() + " instances, " +
                           "and test set, " + _test.x.rows() + " instances.");
    }

    public static void main(String[] args)
        throws IOException
    {
         new jcp_train().run(args);
    }
}
