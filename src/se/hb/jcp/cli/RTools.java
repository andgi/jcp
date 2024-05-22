// JCP - Java Conformal Prediction framework
// Copyright (C) 2024
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
            throws IOException {
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
            throws IOException {
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

        for (int i = 0; i < noPredictions; i++) {
            double lowerBound = predictions[i][0];
            double upperBound = predictions[i][1];

            if (jsonOutput != null) {
                jsonOutput.object();
                jsonOutput.key("instance").value(i);
                jsonOutput.key("true_value").value(testSet.y[i]);
                jsonOutput.key("prediction").value((lowerBound + upperBound) / 2);
                jsonOutput.key("lower_bound").value(lowerBound);
                jsonOutput.key("upper_bound").value(upperBound);
                jsonOutput.endObject();
            }

            if (lowerBound <= testSet.y[i] && testSet.y[i] <= upperBound) {
                correct++;
            }
        }
        long t3 = System.currentTimeMillis();

        if (jsonOutput != null) {
            jsonOutput.endArray();
            jsonOutputBW.close();
        }

        System.out.println("Test Duration " + (double) (t2 - t1) / 1000.0 + " sec.");
        System.out.println("Coverage " + ((double) correct / noPredictions));

        System.out.println("Evaluation Duration " + (double) (t3 - t2) / 1000.0 + " sec.");
    }

    public static IConformalRegressor loadModel(String filename)
            throws IOException {
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
            throws IOException {
        try (ObjectOutputStream oos =
                     new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(cr);
        }
    }
}
