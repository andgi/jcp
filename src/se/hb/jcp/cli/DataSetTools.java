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
import java.util.TreeSet;
import java.util.Random;

import cern.colt.matrix.DoubleMatrix1D;

import se.hb.jcp.cp.*;
import se.hb.jcp.io.*;

/**
 * Higher-level tools for DataSets.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class DataSetTools
{
    public static DataSet loadDataSet(String filename)
        throws IOException
    {
        return
            loadDataSet(filename,
                        new cern.colt.matrix.impl.SparseDoubleMatrix1D(0));
    }

    public static DataSet loadDataSet(String filename,
                                      IConformalClassifier cc)
        throws IOException
    {
        boolean hasClassifier =
            cc != null && cc.getNonconformityFunction().getClassifier() != null;
        DoubleMatrix1D template =
            hasClassifier
            ? cc.getNonconformityFunction().getClassifier().
                  nativeStorageTemplate()
            : new cern.colt.matrix.impl.SparseDoubleMatrix1D(0);

        DataSet dataSet = loadDataSet(filename, template);

        if (hasClassifier &&
            cc.getNonconformityFunction().getClassifier().
                getAttributeCount() > -1 &&
            dataSet.x.columns() !=
                cc.getNonconformityFunction().getClassifier().
                    getAttributeCount()) {
            System.err.println
                ("Warning: " +
                 "The number of attributes in the data set, " +
                 dataSet.x.columns() + ", " +
                 "does not match the number of attributes in the model, " +
                 cc.getNonconformityFunction().getClassifier().
                     getAttributeCount() + ".");
        }
        return dataSet;
    }

    public static DataSet loadDataSet(String filename,
                                      DoubleMatrix1D template)
        throws IOException
    {
        FileInputStream file;
        file = new FileInputStream(filename);
        DataSet dataSet = new libsvmReader().read(file, template);
        file.close();

        System.out.println("Loaded the dataset " + filename + " containing " +
                           dataSet.x.rows() + " instances with " +
                           dataSet.x.columns() + " attributes.");
        return dataSet;
    }

    public static
        SimpleEntry<double[],SortedSet<Double>> extractClasses(DataSet dataSet)
    {
        TreeSet<Double> classSet = new TreeSet<Double>();
        for (int r = 0; r < dataSet.x.rows(); r++) {
            if (!classSet.contains(dataSet.y[r])) {
                classSet.add(dataSet.y[r]);
            }
        }
        double[] classes = new double[classSet.size()];
        System.out.println("Classes: ");
        int i = 0;
        for (Double c : classSet.toArray(new Double[0])) {
            classes[i] = c;
            System.out.println("   Class " + i + " with label '" + classes[i] +
                               "'.");
            i++;
        }
        return new SimpleEntry<double[],SortedSet<Double>>(classes, classSet);
    }
}
