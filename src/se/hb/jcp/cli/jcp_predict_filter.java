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

import java.util.Iterator;
import java.util.Scanner;

import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.json.JSONWriter;

import se.hb.jcp.cp.*;

/**
 * Command line filter for making predictions for JSON formatted instances
 * read from stdin and write the JSON formatted predictions to stdout.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class jcp_predict_filter
{
    String _modelFileName;

    public jcp_predict_filter()
    {
    }

    public void run(String[] args)
        throws IOException
    {
        processArguments(args);

        doClassification();
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
                } else {
                    // The last unknown argument should be the dataset file.
                    _modelFileName = args[i];
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
    }

    private void printUsage()
    {
        System.out.println
            ("Usage: jcp_predict_filter [options] <model file>");
        System.out.println();
        System.out.println
            ("  -h                Print this message and exit.");
    }

    private void doClassification()
        throws IOException
    {
        IConformalClassifier cc = CCTools.loadModel(_modelFileName);
        JSONTokener instanceReader = new JSONTokener(System.in);
        OutputStreamWriter osw     = new OutputStreamWriter(System.out);
        JSONWriter  resultWriter   = new JSONWriter(osw);

        resultWriter.array();
        DoubleMatrix1D instance = allocateInstance(cc);
        try {
            while (!instanceReader.end()) {
                if (IOTools.readInstanceFromJSON(instanceReader, instance)) {
                    classifyInstance(cc, instance, resultWriter);
                    osw.flush();
                }
            }
        } finally {
            resultWriter.endArray();
            osw.flush();
        }
    }

    private DoubleMatrix1D allocateInstance(IConformalClassifier cc)
    {
        DoubleMatrix1D instance =
            cc.nativeStorageTemplate().like(cc.getAttributeCount());
        return instance;
    }

    private void classifyInstance(IConformalClassifier cc,
                                  DoubleMatrix1D       instance,
                                  JSONWriter           resultWriter)
    {
        // Do the prediction.
        ConformalClassification prediction = cc.predict(instance);
        // Write the result as a JSON object.
        IOTools.writeAsJSON(prediction, resultWriter);
    }

    public static void main(String[] args)
        throws IOException
    {
        new jcp_predict_filter().run(args);
    }
}
