// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.cli;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
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
 * Command line prediction tool for JCP.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class jcp_predict
{
    private String  _testSetFileName;
    private String  _modelFileName;
    private double  _significanceLevel = 0.10;

    private SortedSet<Double> _classSet;
    private double[]          _classes;

    public jcp_predict()
    {
        super();
    }

    public void run(String[] args)
        throws IOException
    {
        processArguments(args);

        // FIXME: Only initial use case yet. Load & test.        
        runICCTest(_modelFileName, _testSetFileName);
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
                } else {
                    // The last unknown argument should be the test set file.
                    _testSetFileName = args[i];
                }
            }
        }
        if (_modelFileName == null) {
            System.err.println
                ("Error: No model file name given.");
            System.err.println();
            printUsage();
            System.exit(-1);
        }
        if (_testSetFileName == null) {
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
            ("Usage: jcp_predict [options] <libsvm formatted data set>");
        System.out.println();
        System.out.println
            ("  -h                Print this message and exit.");
        System.out.println
            ("  -m <model file>   The model to test.");
        System.out.println
            ("  -s <significance> Set the conformal prediction " +
             "significance level for the test phase (0.0-1.0).");
    }

    private void runICCTest(String modelFileName, String dataSetFileName)
        throws IOException
    {
        System.out.println("Loading the model '" + modelFileName +
                           "'.");
        long t1 = System.currentTimeMillis();
        InductiveConformalClassifier icc = loadModel(modelFileName);
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");
                          
        System.out.println("Loading the data set '" + dataSetFileName +
                           "'.");
        DataSet testSet = loadDataset(dataSetFileName);
        extractClasses(testSet);
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        System.out.println("Testing accuracy on " + testSet.x.rows() +
                           " instances at a significance level of " +
                           _significanceLevel + ".");

        // Evaluation on the test set.
        ObjectMatrix2D pred = null;
        pred = icc.predict(testSet.x, _significanceLevel);
        //System.out.println(pred);

        int correct = 0;
        int[] correctAtSize = new int[_classSet.size()+1];
        int[] predictionAtSize = new int[_classSet.size()+1];

        for (int i = 0; i < pred.rows(); i++){
            int classIndex = _classSet.headSet(testSet.y[i]).size();
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
        long t4 = System.currentTimeMillis();

        System.out.println("Accuracy " + ((double)correct / testSet.y.length));
        for (int s = 0; s < predictionAtSize.length; s++) {
            System.out.println("  #Predictions with " + s + " classes: " +
                               predictionAtSize[s] + ". Accuracy: " +
                               (double)correctAtSize[s]/(double)predictionAtSize[s]);
        }
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");
        System.out.println("Total Duration " + (double)(t4 - t1)/1000.0 +
                           " sec.");
    }

    private DataSet loadDataset(String filename)
        throws IOException
    {
        FileInputStream file;
        file = new FileInputStream(filename);

        DataSet dataSet =
            new libsvmReader().read
                (file,
                 // Select the desired representation.
                 //new cern.colt.matrix.impl.SparseDoubleMatrix2D(0, 0));
                 new jcp.bindings.libsvm.SparseDoubleMatrix2D(0, 0));
        file.close();

        System.out.println("Loaded the dataset " + filename + " containing " +
                           dataSet.x.rows() + " instances with " +
                           dataSet.x.columns() + " attributes.");
        return dataSet;
    }

    private void extractClasses(DataSet dataSet)
    {
        _classSet = new TreeSet<Double>();
        for (int r = 0; r < dataSet.x.rows(); r++) {
            if (!_classSet.contains(dataSet.y[r])) {
                _classSet.add(dataSet.y[r]);
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

    private InductiveConformalClassifier loadModel(String filename)
        throws IOException
    {
        FileInputStream   file;
        ObjectInputStream ois;
        InductiveConformalClassifier icc = null;
        file = new FileInputStream(filename);
        ois  = new ObjectInputStream(file);

        try {
            icc = (InductiveConformalClassifier)ois.readObject();
        } catch (Exception e) {
            throw new IOException("Failed to load ICC model from '" +
                                  filename + "'. " + e.getMessage());
        }
        ois.close();
        file.close();
        return icc;
    }

    public static void main(String[] args)
        throws IOException
    {
         new jcp_predict().run(args);
    }
}
