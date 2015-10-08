// JCP - Java Conformal Prediction framework
// Copyright (C) 2015  Anders Gidenstam
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
package se.hb.jcp.cli;

import java.io.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.SortedSet;

import cern.colt.matrix.DoubleMatrix2D;

import se.hb.jcp.cp.*;
import se.hb.jcp.nc.*;
import se.hb.jcp.io.*;

/**
 * Higher-level tools for Conformal Classification.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class CCTools
{
    public static void runTest(String modelFileName,
                               String dataSetFileName,
                               String pValuesOutputFileName,
                               String labelsOutputFileName,
                               double significanceLevel)
        throws IOException
    {
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

        runTest(cc, testSet, pValuesOutputFileName,
                labelsOutputFileName, significanceLevel);

        long t4 = System.currentTimeMillis();
        System.out.println("Total Duration " + (double)(t4 - t1)/1000.0 +
                           " sec.");
    }

    public static void runTest(IConformalClassifier cc,
                               DataSet testSet,
                               String pValuesOutputFileName,
                               String labelsOutputFileName,
                               double significanceLevel)
        throws IOException
    {
        BufferedWriter pValuesOutput = null;
        if (pValuesOutputFileName != null) {
            pValuesOutput =
                new BufferedWriter
                    (new OutputStreamWriter
                        (new FileOutputStream(pValuesOutputFileName), "utf-8"));
        }
        BufferedWriter labelsOutput = null;
        if (labelsOutputFileName != null) {
            labelsOutput =
                new BufferedWriter
                    (new OutputStreamWriter
                        (new FileOutputStream(labelsOutputFileName), "utf-8"));
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
        DoubleMatrix2D pValues = cc.predictPValues(testSet.x);

        int correct = 0;
        int[] correctAtSize = new int[classSet.size()+1];
        int[] predictionAtSize = new int[classSet.size()+1];

        for (int i = 0; i < pValues.rows(); i++){
            int classIndex = classSet.headSet(testSet.y[i]).size();
            int predictionSize = 0;
            for (int c = 0; c < classes.length; c++) {
                double pValue = pValues.get(i, c);
                if (pValuesOutput != null) {
                    pValuesOutput.write("" + pValue + " ");
                }
                if (pValue >= significanceLevel) {
                    // This label cannot be excluded.
                    predictionSize++;
                    if (labelsOutput != null) {
                        labelsOutput.write("" + classes[c] + " ");
                    }
                }
            }
            if (pValuesOutput != null) {
                pValuesOutput.newLine();
            }
            if (labelsOutput != null) {
                labelsOutput.newLine();
            }

            predictionAtSize[predictionSize]++;

            if (pValues.get(i, classIndex) >= significanceLevel) {
                correct++;
                correctAtSize[predictionSize]++;
            }
        }
        long t3 = System.currentTimeMillis();

        if (pValuesOutput != null) {
            pValuesOutput.close();
        }
        if (labelsOutput != null) {
            labelsOutput.close();
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
