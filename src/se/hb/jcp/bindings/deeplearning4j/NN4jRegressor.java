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
// The public interface is based on cern.colt.matrix.DoubleMatrix2D.
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

import org.json.JSONObject;

import se.hb.jcp.ml.RegressorBase;
import se.hb.jcp.ml.IRegressor;

public class NN4jRegressor extends RegressorBase implements IRegressor, java.io.Serializable {
    private static final SparseDoubleMatrix1D _storageTemplate = new SparseDoubleMatrix1D(0);
    protected MultiLayerNetwork _model;

    public NN4jRegressor() {}

    public NN4jRegressor(JSONObject configuration) {
        this();
        _model = createNetworkFromConfig(configuration);
    }

    public NN4jRegressor(String modelFilePath) {
        this();
        try {
            _model = MultiLayerNetwork.load(new File(modelFilePath), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public NN4jRegressor(MultiLayerNetwork model) {
        _model = model;
    }

    @Override
    public IRegressor fitNew(DoubleMatrix2D x, double[] y) {
        internalFit(x, y);
        return new NN4jRegressor(_model);
    }

    @Override
    public double predict(DoubleMatrix1D instance) {
        INDArray input = Nd4j.create(instance.toArray()).reshape(1, instance.size());
        INDArray output = _model.output(input);
        System.out.println("Prediction " + output);
        return output.getDouble(0);
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        return _storageTemplate;
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) {
        _model = createAndTrainNetwork(x, y, 50);
    }

    private MultiLayerNetwork createAndTrainNetwork(DoubleMatrix2D x, double[] y, int hiddenLayerSize) {
        int inputFeatures = x.columns();
        int nEpochs = 50;
        double learningRate = 0.0001;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
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
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        DataSet dataSet = createDataSet(x, y);
        for (int i = 0; i < nEpochs; i++) {
            model.fit(dataSet);
        }

        return model;
    }

    private DataSet createDataSet(DoubleMatrix2D x, double[] y) {
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
    public NN4jRegressor clone() {
        try {
            // Serialize and then deserialize to achieve deep cloning
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

    private MultiLayerNetwork createNetworkFromConfig(JSONObject config) {
        return null;
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        if (_model != null) {
            String fileName = Long.toHexString(Double.doubleToLongBits(Math.random())) + ".deeplearning4j";
            _model.save(new File(fileName));
            oos.writeObject(fileName);
        } else {
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        String fileName = (String) ois.readObject();
        if (fileName != null) {
            _model = MultiLayerNetwork.load(new File(fileName), true);
        }
    }
}
