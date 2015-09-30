// Copyright (C) 2014 - 2015  Anders Gidenstam
// License: to be defined.
package jcp.cli;

import java.io.*;

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
    private String  _labelsOutputFileName;
    private String  _pValuesOutputFileName;
    private String  _modelFileName;
    private double  _significanceLevel = 0.10;

    public jcp_predict()
    {
        super();
    }

    public void run(String[] args)
        throws IOException
    {
        processArguments(args);

        CCTools.runTest(_modelFileName, _testSetFileName,
                        _pValuesOutputFileName, _labelsOutputFileName,
                        _significanceLevel);
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
                } else if (args[i].equals("-sl")) {
                    if (++i < args.length) {
                        _labelsOutputFileName = args[i];
                    } else {
                        System.err.println
                            ("Error: No file name given to -sl.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-sp")) {
                    if (++i < args.length) {
                        _pValuesOutputFileName = args[i];
                    } else {
                        System.err.println
                            ("Error: No file name given to -sp.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].startsWith("-")) {
                    System.err.println
                        ("Error: Unknown option '" + args[i] + "'.");
                    System.err.println();
                    printUsage();
                    System.exit(-1);
                } else {
                    // Any unrecognized argument should be the test set file.
                    if (_testSetFileName == null) {
                        _testSetFileName = args[i];
                    } else {
                        System.err.println
                            ("Error: Unexpected redundant argument found '" +
                             args[i] + "'.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
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
             "significance level (0.0-1.0).");
        System.out.println
            ("  -sl <file>        Save the predicted labels in <file>.");
        System.out.println
            ("  -sp <file>        Save the predicted p-values in <file>.");
    }

    public static void main(String[] args)
        throws IOException
    {
         new jcp_predict().run(args);
    }
}
