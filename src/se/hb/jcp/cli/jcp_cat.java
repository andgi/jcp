// JCP - Java Conformal Prediction framework
// Copyright (C) 2016  Anders Gidenstam
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

import java.io.IOException;
import java.io.OutputStreamWriter;

import cern.colt.matrix.DoubleMatrix1D;
import org.json.JSONWriter;

import se.hb.jcp.cp.DataSet;

/**
 * Command line tool for converting a data set file to JSON format written
 * on stdout.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class jcp_cat
{
    private String  _dataSetFileName;
    private DataSet _dataSet;
    private boolean _includeTarget = false;

    public jcp_cat()
    {
        _dataSet = new DataSet();
    }

    public void run(String[] args)
        throws IOException
    {
        processArguments(args);

        loadDataSet();
        writeDataSetAsJSON();
    }

    private void processArguments(String[] args)
    {
        if (args.length < 1) {
            printUsage();
            System.exit(-1);
        } else {
            for (int i = 0; i < args.length; i++) {
                if (args[i].equals("-h")) {
                    printUsage();
                    System.exit(0);
                } else if (args[i].equals("-t")) {
                    _includeTarget = true;
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
            ("Usage: jcp_cat [options] <libsvm formatted data set>");
        System.out.println();
        System.out.println
            ("  -h                Print this message and exit.");
        System.out.println
            ("  -t                Include the target as the very last column.");
    }

    private void loadDataSet()
        throws IOException
    {
        long t1  = System.currentTimeMillis();
        _dataSet = DataSetTools.loadDataSet(_dataSetFileName);
        long t2  = System.currentTimeMillis();
        System.err.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");
    }

    private void writeDataSetAsJSON()
        throws IOException
    {
        OutputStreamWriter osw = new OutputStreamWriter(System.out);
        JSONWriter jsonWriter  = new JSONWriter(osw);
        jsonWriter.array();
        for (int i = 0; i < _dataSet.x.rows(); i++) {
            DoubleMatrix1D instance = _dataSet.x.viewRow(i);
            if (_includeTarget) {
                IOTools.writeAsJSON(instance, _dataSet.y[i], jsonWriter);
            } else {
                IOTools.writeAsJSON(instance, jsonWriter);
            }
        }
        jsonWriter.endArray();
        osw.flush();
    }

    public static void main(String[] args)
        throws IOException
    {
         new jcp_cat().run(args);
    }
}
