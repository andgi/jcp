// JCP - Java Conformal Prediction framework
// Copyright (C) 2024  Tom le Cam
// Copyright (C) 2024  Anders Gidenstam
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
package se.hb.jcp.bindings.deeplearning4j;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONArray;
import org.json.JSONObject;

import se.hb.jcp.ml.RegressorBase;
import se.hb.jcp.ml.IRegressor;

public class NN4jRegressor
    extends RegressorBase
    implements IRegressor, java.io.Serializable
{
    private static final SparseDoubleMatrix1D _storageTemplate = new SparseDoubleMatrix1D(0);
    protected JSONObject _jsonParameters;
    protected MultiLayerNetwork _model;

    public NN4jRegressor() {}

    public NN4jRegressor(JSONObject configuration)
    {
        this();
        _model = createNetworkFromConfig(configuration);
    }

    public NN4jRegressor(String modelFilePath)
    {
        this();
        try {
            _model = MultiLayerNetwork.load(new File(modelFilePath), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public NN4jRegressor(MultiLayerNetwork model)
    {
        _model = model;
    }

    @Override
    public IRegressor fitNew(DoubleMatrix2D x, double[] y)
    {
        NN4jRegressor newRegressor = this.clone();
        newRegressor.internalFit(x, y);
        return newRegressor;
    }

    @Override
    public double predict(DoubleMatrix1D instance)
    {
        INDArray input = Nd4j.create(instance.toArray()).reshape(1, instance.size());
        INDArray output = _model.output(input);
        //System.out.println("  Prediction " + output);
        return output.getDouble(0);
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y)
    {
        _model = createAndTrainNetwork(x, y, 50);
    }

    private MultiLayerNetwork createAndTrainNetwork(DoubleMatrix2D x,
                                                    double[] y,
                                                    int hiddenLayerSize)
    {
        int inputFeatures = x.columns();
        int nEpochs = 50;
        double learningRate = 0.0001;

        NNParameters parameters = readJSONParameters(inputFeatures);
        // FIXME
        parameters.println();

        MultiLayerConfiguration conf =
            new NeuralNetConfiguration.Builder()
                .seed(parameters.seed)
                .activation(parameters.activation)
                .weightInit(parameters.weightInit)
                .updater(new Nesterovs(parameters.updater.learningRate,
                                       parameters.updater.momentum))
                .l2(parameters.l2)
                .list()
                .layer(new DenseLayer.Builder()
                           .nIn(inputFeatures)
                           .nOut(parameters.layers.get(0).out).build())
                .layer(new DenseLayer.Builder()
                           .nIn(parameters.layers.get(1).in)
                           .nOut(parameters.layers.get(1).out).build())
                .layer(new DenseLayer.Builder()
                           .nIn(parameters.layers.get(2).in)
                           .nOut(parameters.layers.get(2).out).build())
                .layer(new OutputLayer.Builder(parameters.layers.get(3).lossFunction)
                    .activation(parameters.layers.get(3).activation)
                    .nIn(parameters.layers.get(3).in)
                    .nOut(parameters.layers.get(3).out).build())
                .build();

/*            .seed(12345)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(learningRate, 0.9))
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(inputFeatures).nOut(hiddenLayerSize).build())
            .layer(new DenseLayer.Builder().nIn(50).nOut(hiddenLayerSize).build())
            .layer(new DenseLayer.Builder().nIn(50).nOut(hiddenLayerSize).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(hiddenLayerSize).nOut(1).build())
            .build();*/
        // End FIXME

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        DataSet dataSet = createDataSet(x, y);
        for (int i = 0; i < nEpochs; i++) {
            model.fit(dataSet);
        }

        return model;
    }

    private DataSet createDataSet(DoubleMatrix2D x, double[] y)
    {
        int rows = x.rows();
        int cols = x.columns();
        INDArray features = Nd4j.create(x.toArray());
        INDArray labels = Nd4j.create(y, new int[]{rows, 1});
        DataSet dataSet = new DataSet(features, labels);
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);
        return dataSet;
    }

    @Override
    public NN4jRegressor clone()
    {
        try {
            // Serialize and then deserialize to achieve deep cloning.
            ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
            ObjectOutputStream out = new ObjectOutputStream(byteOut);
            out.writeObject(this);
            ByteArrayInputStream byteIn = new ByteArrayInputStream(byteOut.toByteArray());
            ObjectInputStream in = new ObjectInputStream(byteIn);
            return (NN4jRegressor) in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    private MultiLayerNetwork createNetworkFromConfig(JSONObject config)
    {
        return null;
    }

    private NNParameters readJSONParameters(int inputFeatures)
    {
        // FIXME
        int hiddenLayerSize = 50;
        // End FIXME
        if (_jsonParameters == null) {
            String JSON =
                "{\"activation\":\"RELU\"," +
                "\"l2\":0.000000000001," +
                "\"seed\":42," +
                "\"weight_init\":\"XAVIER\"," +
                "\"updater\":{" +
                "  \"name\":\"NESTEROVS\"," +
                "  \"learning_rate\":0.00001," +
                "  \"momentum\":0.91}," +
                "\"layers\":["+
                "    {\"input_size\":" + inputFeatures + "," +
                "     \"output_size\":" + hiddenLayerSize + "" +
                "    }," +
                "    {" +
                "     \"output_size\":" + hiddenLayerSize + "" +
                "    }," +
                "    {" +
                "     \"output_size\":" + hiddenLayerSize + "" +
                "    }," +
                "    {" +
                "     \"loss_function\":\"MSE\"," +
                "     \"activation\":\"IDENTITY\"," +
                "     \"output_size\":1" +
                "    }" +
                "  ]" +
                "}";
            System.out.println("EXPECTED JSON: " + JSON);
            _jsonParameters = new JSONObject(JSON);
        }
        NNParameters parameters = new NNParameters();
        if (_jsonParameters != null) {
            if (_jsonParameters.has("activation")) {
                String type = _jsonParameters.getString("activation");
                if (type.equals("CUBE")) {
                    parameters.activation = Activation.CUBE;
                } else if (type.equals("ELU")) {
                    parameters.activation = Activation.ELU;
                } else if (type.equals("RELU")) {
                    parameters.activation = Activation.RELU;
                } else if (type.equals("SIGMOID")) {
                    parameters.activation = Activation.SIGMOID;
                } else if (type.equals("TANH")) {
                    parameters.activation = Activation.TANH;
                } else {
                    throw new IllegalArgumentException
                                  ("se.hb.jcp.bindings.deeplearning4j.NN4jRegressor:"
                                   + "Unknown activation type '" + type + "'.");
                }
            }
            if (_jsonParameters.has("l2")) {
                parameters.l2 = _jsonParameters.getDouble("l2");
            }
            if (_jsonParameters.has("layers")) {
                // "layers":[{},{},...,{}]
                // layer: {in, out, activation, loss-function, ...}
                JSONArray jsonLayers = _jsonParameters.getJSONArray("layers");
                for (int i = 0; i < jsonLayers.length(); ++i) {
                    int in;
                    int out;
                    if (i == 0) {
                        in = inputFeatures;
                    } else {
                        in = jsonLayers.getJSONObject(i-1).getInt("output_size");
                    }
                    out = jsonLayers.getJSONObject(i).getInt("output_size");
                    if (jsonLayers.getJSONObject(i).has("activation") &&
                        jsonLayers.getJSONObject(i).has("loss_function")) {
                        Activation act = null;
                        LossFunctions.LossFunction lossFunc = null;
                        String type =
                            jsonLayers.getJSONObject(i).getString("activation");
                        if (type.equals("IDENTITY")) {
                            act = Activation.IDENTITY;
                        } else {
                            throw new IllegalArgumentException
                                          ("se.hb.jcp.bindings.deeplearning4j.NN4jRegressor:"
                                           + "Unknown activation type '" + type + "'.");
                        }
                        type =
                            jsonLayers.getJSONObject(i).getString("loss_function");
                        if (type.equals("MSE")) {
                            lossFunc = LossFunctions.LossFunction.MSE;
                        } else {
                            throw new IllegalArgumentException
                                          ("se.hb.jcp.bindings.deeplearning4j.NN4jRegressor:"
                                           + "Unknown loss-function type '" + type + "'.");
                        }
                        parameters.layers.add(new NNLayer(in, out,
                                                          act, lossFunc));
                    } else {
                        parameters.layers.add(new NNLayer(in, out));
                    }
                }
            }
            if (_jsonParameters.has("seed")) {
                parameters.seed = _jsonParameters.getInt("seed");
            }
            if (_jsonParameters.has("weight_init")) {
                String type = _jsonParameters.getString("weight_init");
                if (type.equals("XAVIER")) {
                    parameters.weightInit = WeightInit.XAVIER;
                } else {
                    throw new IllegalArgumentException
                                  ("se.hb.jcp.bindings.deeplearning4j.NN4jRegressor:"
                                   + "Unknown weight initializtion type '"
                                   + type + "'.");
                }
            }
            if (_jsonParameters.has("updater")) {
                JSONObject jsonUpdater = _jsonParameters.getJSONObject("updater");
                // FIXME: name, learning_rate, others...
                // name: SGD, ADAM, ADAMAX, ADADELTA, NESTEROVS, NADAM, ADAGRAD,
                //       RMSPROP
                String jsonName = "";
                if (jsonUpdater.has("name")) {
                    jsonName = jsonUpdater.getString("name");
                    if (jsonName.equals("SGD")) {
                    } else if (jsonName.equals("ADAM")) {
                    } else if (jsonName.equals("ADAMAX")) {
                    } else if (jsonName.equals("ADADELTA")) {
                    } else if (jsonName.equals("NESTEROVS")) {
                    } else if (jsonName.equals("NADAM")) {
                    } else if (jsonName.equals("ADAGRAD")) {
                    } else if (jsonName.equals("RMSPROP")) {
                    } else {
                        throw new IllegalArgumentException
                                      ("se.hb.jcp.bindings.deeplearning4j.NN4jRegressor:"
                                       + "Unknown updater '"
                                       + jsonName + "'.");
                    }
                } else {
                    throw new IllegalArgumentException
                                  ("se.hb.jcp.bindings.deeplearning4j.NN4jRegressor:"
                                   + "Unknown updater '"
                                   + jsonName + "'.");
                }
                double learningRate = 0.0001;
                if (jsonUpdater.has("learning_rate")) {
                    learningRate = jsonUpdater.getDouble("learning_rate");
                }
                double momentum = 0.9;
                if (jsonUpdater.has("momentum")) {
                    momentum = jsonUpdater.getDouble("momentum");
                }
                parameters.updater =
                    new NNUpdater(jsonName, learningRate, momentum);
            }
        }
        return parameters;
    }

    private void writeObject(ObjectOutputStream oos)
        throws IOException
    {
        if (_model != null) {
            String fileName = Long.toHexString(Double.doubleToLongBits(Math.random())) + ".deeplearning4j";
            _model.save(new File(fileName));
            oos.writeObject(fileName);
        } else {
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, IOException
    {
        String fileName = (String) ois.readObject();
        if (fileName != null) {
            _model = MultiLayerNetwork.load(new File(fileName), true);
        }
    }

    private class NNParameters
    {
        Activation activation = null;
        double l2 = 0.0;
        ArrayList<NNLayer> layers = new ArrayList<NNLayer>();
        int seed = 0;
        WeightInit weightInit = null;
        NNUpdater updater = null;

        void println()
        {
            System.out.println("READ PARAMETERS: ");
            System.out.println("  Activation = " + activation);
            System.out.println("  L2 = " + l2);
            System.out.println("  Layers = ");
            for (int i = 0; i < layers.size(); ++i) {
                NNLayer layer = layers.get(i);
                System.out.println("    Layer " + i + " {" +
                                   " in = " + layer.in +
                                   " out = " + layer.out +
                                   " activation = " + layer.activation +
                                   " lossFunction = " + layer.lossFunction +
                                   " }");
            }
            System.out.println("  Seed = " + seed);
            System.out.println("  WeightInit = " + weightInit);
            System.out.println("  Updater = { " +
                               " name = " + updater.name +
                               " learningRate = " + updater.learningRate +
                               " momentum = " + updater.momentum +
                               " }");
        }
    }

    private class NNLayer
    {
        int in;
        int out;
        Activation activation = null;
        LossFunctions.LossFunction lossFunction = null;

        NNLayer(int in, int out)
        {
            this.in = in;
            this.out = out;
        }
        NNLayer(int in, int out,
                Activation activation, LossFunctions.LossFunction lossFunction)
        {
            this(in, out);
            this.activation = activation;
            this.lossFunction = lossFunction;
        }
    }

    private class NNUpdater
    {
        String name;
        double learningRate;
        double momentum;

        NNUpdater(String name, double learningRate, double momentum)
        {
            this.name = name;
            this.learningRate = learningRate;
            this.momentum = momentum;
        }
    }
}
