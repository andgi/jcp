// JCP - Java Conformal Prediction framework
// Copyright (C) 2024  Tom le Cam
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
import java.util.Arrays;
import org.json.JSONWriter;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.DoubleMatrix1D;
import se.hb.jcp.cp.*;

/**
 * Higher-level tools for Conformal Regression.
 */
public class RTools {

    public static void runTest(String modelFileName,
                               String dataSetFileName,
                               String jsonOutputFileName,
                               double confidenceLevel,
                               boolean debug)
        throws IOException
    {
        System.out.println("Loading the model '" + modelFileName +
                           "'.");
        long t1 = System.currentTimeMillis();
        IConformalRegressor cr = loadModel(modelFileName);
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double) (t2 - t1) / 1000.0 + " sec.");

        System.out.println("Loading the data set '" + dataSetFileName +
                           "'.");
        DataSet testSet = DataSetTools.loadDataSet(dataSetFileName, cr);
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double) (t3 - t2) / 1000.0 + " sec.");

        runTest(cr, testSet, jsonOutputFileName, confidenceLevel, debug);
        long t4 = System.currentTimeMillis();
        System.out.println("Total Duration " + (double) (t4 - t1) / 1000.0 +
                           " sec.");
    }

    public static void runTest(IConformalRegressor cr,
                           DataSet testSet,
                           String jsonOutputFileName,
                           double confidenceLevel,
                           boolean debug)
        throws IOException
    {
        BufferedWriter jsonOutputBW = null;
        JSONWriter jsonOutput = null;
        if (jsonOutputFileName != null) {
            jsonOutputBW =
                    new BufferedWriter(
                            new OutputStreamWriter(
                                    new FileOutputStream(jsonOutputFileName), "utf-8"));
            jsonOutput = new JSONWriter(jsonOutputBW);
            jsonOutput.array();
        }

        long t1 = System.currentTimeMillis();
        System.out.println("Testing on " + testSet.x.rows() +
                           " instances at a confidence level of " +
                           confidenceLevel + ".");

        // Evaluation on the test set.
        double[][] predictions = cr.predictIntervals(testSet.x, confidenceLevel);
        long t2 = System.currentTimeMillis();

        int noPredictions = testSet.y.length;
        int correct = 0;
        int nearlyPerfectPred = 0;
        double sumAbsoluteError = 0.0;
        double sumSquaredError = 0.0;
        //Efficiency of a conformal regression (see Report series/DSV - On effectively creating ensembles of classifiers - 3.1.1 page 42)
        double sumIntervalWidth = 0.0;
        double[] intervalWidths = new double[noPredictions];

        for (int i = 0; i < noPredictions; i++) {
            double lowerBound = predictions[i][0];
            double upperBound = predictions[i][1];
            double intervalWidth = upperBound - lowerBound;
            sumIntervalWidth += intervalWidth;
            intervalWidths[i] = intervalWidth;
            double predictedValue = (lowerBound + upperBound) / 2;


            if (jsonOutput != null) {
                jsonOutput.object();
                jsonOutput.key("instance").value(i);
                jsonOutput.key("true_value").value(testSet.y[i]);
                jsonOutput.key("prediction").value(predictedValue);
                jsonOutput.key("lower_bound").value(lowerBound);
                jsonOutput.key("upper_bound").value(upperBound);
                jsonOutput.endObject();
            }

            double trueValue = testSet.y[i];

            double tolerance = 1.0;

            if (trueValue <= predictedValue + tolerance &&
                trueValue >= predictedValue - tolerance) {
                nearlyPerfectPred++;
            }
            if (lowerBound <= trueValue && trueValue <= upperBound) {
                correct++;
            } else {
                System.out.println("Lower bound :" + lowerBound + " Upper bound : " + upperBound + " Predicted value " + predictedValue + " True value : " + testSet.y[i]);

            }

            double error = trueValue - predictedValue;
            sumAbsoluteError += Math.abs(error);
            sumSquaredError += error * error;
        }
        long t3 = System.currentTimeMillis();

        if (jsonOutput != null) {
            jsonOutput.endArray();
            jsonOutputBW.close();
        }

        double averageIntervalWidth = sumIntervalWidth / noPredictions;
        Arrays.sort(intervalWidths);
        double medianIntervalWidth = calculateMedian(intervalWidths);
        double mae = sumAbsoluteError / noPredictions;
        double mse = sumSquaredError / noPredictions;
        double rmse = Math.sqrt(mse);
        double coverage = (double) correct / noPredictions;

        double perfectRatio = (double) nearlyPerfectPred / noPredictions;

        System.out.println("Test Duration " + (double) (t2 - t1) / 1000.0 + " sec.");
        System.out.println("Coverage " + coverage);
        System.out.println("Perfect prediction ratio " + perfectRatio);
        System.out.println("Mean Absolute Error (MAE): " + mae);
        System.out.println("Mean Squared Error (MSE): " + mse);
        System.out.println("Root Mean Squared Error (RMSE): " + rmse);
        System.out.println("Efficiency : Average Interval Width: " + averageIntervalWidth);
        System.out.println("Efficiency : Median Interval Width: " + medianIntervalWidth);
        System.out.println("Evaluation Duration " + (double) (t3 - t2) / 1000.0 + " sec.");
    }

    private static double calculateMedian(double[] arr)
    {
        int len = arr.length;
        if (len % 2 == 0) {
            return (arr[len / 2 - 1] + arr[len / 2]) / 2.0;
        } else {
            return arr[len / 2];
        }
    }

    public static IConformalRegressor loadModel(String filename)
        throws IOException
    {
        IConformalRegressor cr = null;

        try (ObjectInputStream ois =
                     new ObjectInputStream(new FileInputStream(filename))) {
            cr = (IConformalRegressor) ois.readObject();
        } catch (Exception e) {
            throw new IOException("Failed to load Conformal Regressor model" +
                                  " from '" +
                                  filename + "'.\n" +
                                  e + "\n" +
                                  e.getMessage() + "\n" +
                                  e.getStackTrace());
        }
        return cr;
    }

    public static void saveModel(IConformalRegressor cr,
                                 String filename)
        throws IOException
    {
        try (ObjectOutputStream oos =
                     new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(cr);
        }
    }
}
