// JCP - Java Conformal Prediction framework
// Copyright (C) 2015 - 2016, 2019  Anders Gidenstam
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
import java.util.TreeSet;

import cern.colt.matrix.DoubleMatrix1D;

import se.hb.jcp.cp.*;
import se.hb.jcp.io.*;
import se.hb.jcp.ml.IClassifier;

/**
 * Higher-level tools for DataSets.
 */
public class DataSetTools {

    public static DataSet loadDataSet(String filename)
            throws IOException {
        return loadDataSet(filename,
                new cern.colt.matrix.impl.SparseDoubleMatrix1D(0));
    }

    public static DataSet loadDataSet(String filename,
                                      IConformalClassifier cc)
            throws IOException {
        DataSet dataSet = loadDataSet(filename, cc.nativeStorageTemplate());

        if (cc.isTrained() && dataSet.x.columns() != cc.getAttributeCount()) {
            System.err.println("Warning: " +
                    "The number of attributes in the data set, " +
                    dataSet.x.columns() + ", " +
                    "does not match the number of attributes in the model, " +
                    cc.getAttributeCount() + ".");
        }
        return dataSet;
    }

    public static DataSet loadDataSet(String filename,
                                      IClassifier classifier)
            throws IOException {
        DataSet dataSet = loadDataSet(filename,
                classifier.nativeStorageTemplate());

        if (classifier.isTrained() &&
                dataSet.x.columns() != classifier.getAttributeCount()) {
            System.err.println("Warning: " +
                    "The number of attributes in the data set, " +
                    dataSet.x.columns() + ", " +
                    "does not match the number of attributes in the model, " +
                    classifier.getAttributeCount() + ".");
        }
        return dataSet;
    }

    public static DataSet loadDataSet(String filename,
                                      IConformalRegressor cr)
            throws IOException {
        DataSet dataSet = loadDataSet(filename, cr.nativeStorageTemplate());

        if (cr.isTrained() && dataSet.x.columns() != cr.getAttributeCount()) {
            System.err.println("Warning: " +
                    "The number of attributes in the data set, " +
                    dataSet.x.columns() + ", " +
                    "does not match the number of attributes in the model, " +
                    cr.getAttributeCount() + ".");
        }
        return dataSet;
    }

    public static DataSet loadDataSet(String filename,
                                      DoubleMatrix1D template)
            throws IOException {
        FileInputStream file;
        file = new FileInputStream(filename);
        DataSet dataSet = new libsvmReader().read(file, template);
        file.close();

        System.err.println("Loaded the dataset " + filename + " (" +
                dataSet.x.getClass().getName() + ") containing " +
                dataSet.x.rows() + " instances with " +
                dataSet.x.columns() + " attributes and " +
                dataSet.x.cardinality() + " nonzeros.");
        return dataSet;
    }

    public static SimpleEntry<double[], SortedSet<Double>> extractClasses(DataSet dataSet) {
        TreeSet<Double> classSet = new TreeSet<Double>();
        for (int r = 0; r < dataSet.x.rows(); r++) {
            classSet.add(dataSet.y[r]);
        }
        double[] classes = new double[classSet.size()];
        System.err.println("Classes: ");
        int i = 0;
        for (Double c : classSet.toArray(new Double[0])) {
            classes[i] = c;
            System.err.println("   Class " + i + " with label '" + classes[i] +
                    "'.");
            i++;
        }
        return new SimpleEntry<double[], SortedSet<Double>>(classes, classSet);
    }
}
