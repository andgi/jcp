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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import cern.colt.matrix.DoubleMatrix1D;
import org.json.JSONTokener;
import org.json.JSONWriter;

import se.hb.jcp.cp.ConformalClassification;
import se.hb.jcp.cp.IConformalClassifier;
import se.hb.jcp.util.FIFOParallelExecutor;

/**
 * Command line filter for making predictions for JSON formatted instances
 * read from stdin and write the JSON formatted predictions to stdout.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class jcp_predict_filter
{
    static final boolean PARALLEL = true;
    ExecutorService _executor;
    String _modelFileName;

    public jcp_predict_filter()
    {
        if (PARALLEL) {
            _executor = Executors.newCachedThreadPool();
        }
    }

    public void run(String[] args)
        throws IOException
    {
        processArguments(args);

        doClassification();

        if (_executor != null) {
            _executor.shutdown();
        }
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
        try {
            if (!PARALLEL) {
                DoubleMatrix1D instance = allocateInstance(cc);
                while (!instanceReader.end()) {
                    if (IOTools.readInstanceFromJSON(instanceReader,
                                                     instance)) {
                        // Do the prediction.
                        ConformalClassification prediction =
                            cc.predict(instance);
                        // Write the result as a JSON object.
                        IOTools.writeAsJSON(prediction, resultWriter);
                        osw.flush();
                    }
                }
            } else {
                final FIFOParallelExecutor<ConformalClassification> queue =
                    new FIFOParallelExecutor<>(_executor);

                // Kick-off the consumer.
                Future<Integer> consumer =
                    _executor.submit(new ResultPrinterCallable(queue,
                                                               osw,
                                                               resultWriter));

                // Read instances from stdin and put them in the executor queue.
                while (!instanceReader.end()) {
                    DoubleMatrix1D instance = allocateInstance(cc);
                    if (IOTools.readInstanceFromJSON(instanceReader,
                                                     instance)) {
                        try {
                            queue.submit(new PredictionCallable(cc, instance));
                        } catch (InterruptedException e) {
                            // FIXME: What to do?
                            System.err.println("jcp_predict_filter: " + e);
                        }
                    }
                }
                try {
                    // Tell the consumer to finish via a null object.
                    queue.submit(new Callable<ConformalClassification>() {
                            public ConformalClassification call()
                            {
                                return null;
                            }
                        });
                    // Wait for the consumer to finish.
                    int count = consumer.get();
                    System.err.println("jcp_predict_filter: Finishing after " +
                                       "classifying " + count + " instances.");
                } catch (InterruptedException e) {
                    // FIXME: What to do?
                    System.err.println("jcp_predict_filter: " + e);
                } catch (ExecutionException e) {
                    // FIXME: What to do?
                    System.err.println("jcp_predict_filter: " + e);
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

    private static class PredictionCallable
        implements Callable<ConformalClassification>
    {
        IConformalClassifier _cc;
        private final DoubleMatrix1D _instance;

        public PredictionCallable(IConformalClassifier cc,
                                  DoubleMatrix1D instance)
        {
            _cc = cc;
            _instance = instance;
        }

        @Override
        public ConformalClassification call()
        {
            return _cc.predict(_instance);
        }
    }

    private static class ResultPrinterCallable implements Callable<Integer>
    {
        private final FIFOParallelExecutor<ConformalClassification> _queue;
        private final OutputStreamWriter _osw;
        private final JSONWriter _resultWriter;

        public ResultPrinterCallable
                   (FIFOParallelExecutor<ConformalClassification> queue,
                    OutputStreamWriter osw,
                    JSONWriter resultWriter)
        {
            _queue = queue;
            _osw = osw;
            _resultWriter = resultWriter;
        }

        @Override
        public Integer call()
        {
            int count = 0;

            try {
                ConformalClassification prediction;
                // While there is a next prediction.
                while ((prediction = _queue.take()) != null) {
                    // Write the result as a JSON object.
                    IOTools.writeAsJSON(prediction,
                                        _resultWriter);
                    _osw.flush();
                    count++;
                }
            } catch (InterruptedException e) {
                // FIXME: What to do?
                System.err.println("ResultPrinterCallable: " + e);
            } catch (ExecutionException e) {
                // FIXME: What to do?
                System.err.println("ResultPrinterCallable: " + e);
            } catch (IOException e) {
                // FIXME: What to do?
                System.err.println("ResultPrinterCallable: " + e);
            }
            return count;
        }
    }

    public static void main(String[] args)
        throws IOException
    {
        new jcp_predict_filter().run(args);
    }
}
