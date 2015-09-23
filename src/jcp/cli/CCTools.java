// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.cli;

import java.io.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.SortedSet;
import java.util.Random;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.ObjectMatrix2D;

import jcp.cp.*;
import jcp.nc.*;
import jcp.io.*;

/**
 * Higher-level tools for Conformal Classification.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class CCTools
{
    public static void runTest(String modelFileName,
                               String dataSetFileName,
                               String outputFileName,
                               double significanceLevel)
        throws IOException
    {
        BufferedWriter output = null;
        if (outputFileName != null) {
            output = new BufferedWriter
                (new OutputStreamWriter(new FileOutputStream(outputFileName),
                                        "utf-8"));
        }

        System.out.println("Loading the model '" + modelFileName +
                           "'.");
        long t1 = System.currentTimeMillis();
        IConformalClassifier cc = loadModel(modelFileName);
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");

        System.out.println("Loading the data set '" + dataSetFileName +
                           "'.");
        DataSet testSet = DataSetTools.loadDataSet(dataSetFileName, cc);
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        runTest(cc, testSet, outputFileName, significanceLevel);

        long t4 = System.currentTimeMillis();
        System.out.println("Total Duration " + (double)(t4 - t1)/1000.0 +
                           " sec.");
    }

    public static void runTest(IConformalClassifier cc,
                               DataSet testSet,
                               String outputFileName,
                               double significanceLevel)
        throws IOException
    {
        BufferedWriter output = null;
        if (outputFileName != null) {
            output = new BufferedWriter
                (new OutputStreamWriter(new FileOutputStream(outputFileName),
                                        "utf-8"));
        }

        long t1 = System.currentTimeMillis();
        System.out.println("Extracting the classes from the test set.");
        SimpleEntry<double[],SortedSet<Double>> pair =
            DataSetTools.extractClasses(testSet);
        double[] classes = pair.getKey();
        SortedSet<Double> classSet = pair.getValue();
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");

        System.out.println("Testing accuracy on " + testSet.x.rows() +
                           " instances at a significance level of " +
                           significanceLevel + ".");

        // Evaluation on the test set.
        ObjectMatrix2D pred = null;
        pred = cc.predict(testSet.x, significanceLevel);
        //System.out.println(pred);

        int correct = 0;
        int[] correctAtSize = new int[classSet.size()+1];
        int[] predictionAtSize = new int[classSet.size()+1];

        for (int i = 0; i < pred.rows(); i++){
            int classIndex = classSet.headSet(testSet.y[i]).size();
            int predictionSize = 0;
            for (int c = 0; c < classes.length; c++) {
                if ((Boolean)pred.get(i, c)) {
                    predictionSize++;
                    if (output != null) {
                        output.write("" + classes[c] + " ");
                    }
                }
            }
            if (output != null) {
                output.newLine();
            }

            predictionAtSize[predictionSize]++;

            if ((Boolean)pred.get(i, classIndex)) {
                correct++;
                correctAtSize[predictionSize]++;
            }
        }
        long t3 = System.currentTimeMillis();

        if (output != null) {
            output.close();
        }

        System.out.println("Accuracy " + ((double)correct / testSet.y.length));
        for (int s = 0; s < predictionAtSize.length; s++) {
            System.out.println("  #Predictions with " + s + " classes: " +
                               predictionAtSize[s] + ". Accuracy: " +
                               (double)correctAtSize[s]/(double)predictionAtSize[s]);
        }
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");
    }

    public static IConformalClassifier loadModel(String filename)
        throws IOException
    {
        IConformalClassifier cc = null;

        try (ObjectInputStream ois =
                 new ObjectInputStream(new FileInputStream(filename))) {
            cc = (IConformalClassifier)ois.readObject();
        } catch (Exception e) {
            throw new IOException("Failed to load Conformal Classifier model" +
                                  " from '" +
                                  filename + "'.\n" +
                                  e + "\n" +
                                  e.getMessage() + "\n" +
                                  e.getStackTrace());
        }
        return cc;
    }

    public static void saveModel(IConformalClassifier cc,
                                 String filename)
        throws IOException
    {
        try (ObjectOutputStream oos =
                 new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(cc);
        }
    }
}
