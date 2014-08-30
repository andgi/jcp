// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.cli;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
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

import jcp.ml.IClassifier;

/**
 * Command line training tool for JCP.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class jcp_train
{
    private IClassifier _classifier;
    private String  _dataSetFileName;
    private String  _modelFileName;
    private DataSet _full;
    private SortedSet<Double> _classSet;
    private double[]          _classes;
    private DataSet _training;
    private DataSet _calibration;
    private DataSet _test;
    private boolean _useCP = true;
    private boolean _validate = false;
    private double  _significanceLevel = 0.10;

    public jcp_train()
    {
        super();
        _training    = new DataSet();
        _calibration = new DataSet();
        _test        = new DataSet();
        _classifier = new jcp.bindings.libsvm.SVMClassifier();
    }

    public void run(String[] args)
        throws IOException
    {
        processArguments(args);
        if (_useCP) {
            // Supports train, calibrate and save and/or test.
            runICCTest(_dataSetFileName);
        } else {
            // FIXME: Only initial use case yet. Train & test.
            runPlainSVMTest(_dataSetFileName);
        }
    }

    private void processArguments(String[] args)
    {
        // Load and create training and calibration sets.
        if (args.length < 1) {
            printUsage();
            System.exit(-1);
        } else {
            for (int i = 0; i < args.length; i++) {
                if (args[i].equals("-h")) {
                    printUsage();
                    System.exit(-1);
                } else if (args[i].equals("-m")) {
                    if (++i < args.length) {
                        _modelFileName = args[i];
                    } else {
                        System.err.println
                            ("Error: No model file name given to -m.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-s")) {
                    if (++i < args.length) {
                        boolean ok = false;
                        try {
                            double s = Double.parseDouble(args[i]);
                            if (0.0 <= s && s <= 1.0) {
                                _significanceLevel = s;
                                ok = true;
                            }
                        } catch (Exception e) {
                            // Handled below as ok is false.
                        }
                        if (!ok) {
                            System.err.println
                                ("Error: Illegal significance level '" +
                                 args[i] +
                                 "' given to -s.");
                            System.err.println();
                            printUsage();
                            System.exit(-1);
                        }
                    } else {
                        System.err.println
                            ("Error: No significance level given to -s.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-v")) {
                    _validate = true;
                } else if (args[i].equals("-nocp")) {
                    _useCP = false;
                } else {
                    // The last unknown argument should be the dataset file.
                    _dataSetFileName = args[i];
                }
            }
        }
        if (_dataSetFileName == null) {
            System.err.println
                ("Error: No data set file name given.");
            System.err.println();
            printUsage();
            System.exit(-1);
        }
    }

    private void printUsage()
    {
        System.out.println
            ("Usage: jcp_train [options] <libsvm formatted data set>");
        System.out.println();
        System.out.println
            ("  -h                Print this message and exit.");
        System.out.println
            ("  -m <model file>   Save the created model.");
        System.out.println
            ("  -s <significance> Set the conformal prediction " +
             "significance level for the test phase (0.0-1.0).");
        System.out.println
            ("  -v                Validate the model after training.");
        System.out.println
            ("                    Reserves 50% of the data set for " +
             "validation.");
        System.out.println
            ("  -nocp             Test classification without " +
             "conformal prediction.");
    }

    private void runICCTest(String dataSetFileName)
        throws IOException
    {
        long t1 = System.currentTimeMillis();
        loadDataset(dataSetFileName);
        extractClasses();
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");

        if (_validate) {
            splitDataset(0.4, 0.1);
        } else {
            splitDataset(0.8, 0.2);
        }
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        System.out.println("Training on " + _training.x.rows() +
                           " instances and calibrating on " +
                           _calibration.x.rows() +
                           " instances.");

        InductiveConformalClassifier icc =
            new InductiveConformalClassifier(_classes);
        icc._nc =
            new ClassProbabilityNonconformityFunction(_classes, _classifier);

        icc.fit(_training.x, _training.y, _calibration.x, _calibration.y);
        long t4 = System.currentTimeMillis();
        System.out.println("Training complete.");
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");

        if (_validate) {
            System.out.println("Testing accuracy on " + _test.x.rows() +
                               " instances at a significance level of " +
                               _significanceLevel + ".");

            // Evaluation on the test set.
            ObjectMatrix2D pred = null;
            pred = icc.predict(_test.x, _significanceLevel);
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

            System.out.println("Accuracy " +
                               ((double)correct / _test.y.length));
            for (int s = 0; s < predictionAtSize.length; s++) {
                System.out.println
                    ("  #Predictions with " + s + " classes: " +
                     predictionAtSize[s] + ". Accuracy: " +
                     (double)correctAtSize[s]/(double)predictionAtSize[s]);
            }
            System.out.println("Duration " + (double)(t5 - t4)/1000.0 +
                               " sec.");
            System.out.println("Total Duration " + (double)(t5 - t1)/1000.0 +
                               " sec.");
        }

        if (_modelFileName != null) {
            System.out.println("Saving the model to '" +
                               _modelFileName + "'...");
            saveModel(icc, _modelFileName);
            System.out.println("... Done.");
        }
    }

    private void runPlainSVMTest(String dataSetFileName)
        throws IOException
    {
        long t1 = System.currentTimeMillis();
        loadDataset(dataSetFileName);
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

        _classifier.fit(_training.x, _training.y);
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
                _classifier.predict(_test.x.viewRow(i), probability);
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
        _training.x = _classifier.nativeStorageTemplate().like2D(0,0);
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

    private void saveModel(InductiveConformalClassifier icc, String filename)
        throws IOException
    {
        try (ObjectOutputStream oos =
                 new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(icc);
        }
    }

    public static void main(String[] args)
        throws IOException
    {
         new jcp_train().run(args);
    }
}
