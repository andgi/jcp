package se.hb.jcp.bindings.deeplearning4j;

import org.nd4j.evaluation.classification.EvaluationBinary;
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
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassProbabilityClassifier;
import se.hb.jcp.ml.IClassifier;

public class NN4jClassifier extends ClassifierBase implements IClassProbabilityClassifier, java.io.Serializable {
    private static final SparseDoubleMatrix1D _storageTemplate = new SparseDoubleMatrix1D(0);
    protected MultiLayerNetwork _model;
    private double _threshold;
    public NN4jClassifier() {}

    public NN4jClassifier(JSONObject configuration) {
        this();
        _model = createNetworkFromConfig(configuration);
    }

    public NN4jClassifier(String modelFilePath) {
        this();
        try {
            _model = MultiLayerNetwork.load(new File(modelFilePath), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public NN4jClassifier(MultiLayerNetwork model) {
        _model = model;
    }

    @Override
    public IClassifier fitNew(DoubleMatrix2D x, double[] y) {
        internalFit(x, y);
        return new NN4jClassifier(_model);
    }

    @Override
    public double predict(DoubleMatrix1D instance) {
        INDArray input = Nd4j.create(instance.toArray()).reshape(1, instance.size());
        INDArray output = _model.output(input);
        double probability = output.getDouble(0);
        return (probability >= 0.5) ? 1.0 : -1.0;
    }

    @Override
    public double predict(DoubleMatrix1D instance, double[] probabilityEstimates) {
        INDArray input = Nd4j.create(instance.toArray()).reshape(1, instance.size());
        INDArray output = _model.output(input);
        double probability = output.getDouble(0);
        probabilityEstimates[0] = 1 - probability;
        probabilityEstimates[1] = probability;
        System.out.println(probability);
        return (probability >= 0.5) ? 1.0 : -1.0;
    }

    public double findBestThreshold(DataSet validationData) {
        double bestThreshold = 0.0;
        double bestF1Score = 0.0;
    
        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.1) {
            EvaluationBinary eval = new EvaluationBinary();
    
            INDArray output = _model.output(validationData.getFeatures());
    
            INDArray predictions = Nd4j.create(output.shape());
            for (int i = 0; i < output.length(); i++) {
                predictions.putScalar(i, output.getDouble(i) >= threshold ? 1.0 : -1.0);
            }
    
            eval.eval(validationData.getLabels(), predictions);
    
            double f1Score = eval.averageF1();
    
            if (f1Score > bestF1Score) {
                bestF1Score = f1Score;
                bestThreshold = threshold;
            }
        }
    
        return bestThreshold;
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        return _storageTemplate;
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) {
        _model = createAndTrainNetwork(x, y);

    }

    private MultiLayerNetwork createAndTrainNetwork(DoubleMatrix2D x, double[] y) {
        int inputFeatures = x.columns();
        //for multiclass we could use softmax 
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(inputFeatures).nOut(50).build())
            .layer(new DenseLayer.Builder().nIn(50).nOut(50).build())
            .layer(new DenseLayer.Builder().nIn(50).nOut(50).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                .activation(Activation.SIGMOID)
                .nIn(50).nOut(1).build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        //DataSet dataSet = createDataSetWithOversampling(x, y);
        DataSet dataSet = createDataSet(x, y);
        model.fit(dataSet);
        _model = model;
        System.out.println("THRESHOLD " + findBestThreshold(dataSet));
        _threshold = findBestThreshold(dataSet);
        return model;
    }



    private DataSet createDataSetWithOversampling(DoubleMatrix2D x, double[] y) {
        int rows = x.rows();
        int cols = x.columns();

        INDArray features = Nd4j.create(x.toArray());
        INDArray labels = Nd4j.create(y, new int[]{rows, 1});

        int countClass0 = 0;
        int countClass1 = 0;
        for (double label : y) {
            if (label == -1.0) countClass0++;
            else countClass1++;
        }
        if(countClass0 != 0) {

            int oversampleFactor = (countClass1 / countClass0) - 1;

            List<INDArray> oversampledFeatures = new ArrayList<>();
            List<INDArray> oversampledLabels = new ArrayList<>();

            for (int i = 0; i < rows; i++) {
                oversampledFeatures.add(features.getRow(i));
                oversampledLabels.add(labels.getRow(i));
                if (y[i] == -1.0) {
                    for (int j = 0; j < oversampleFactor; j++) {
                        oversampledFeatures.add(features.getRow(i));
                        oversampledLabels.add(labels.getRow(i));
                    }
                }
            }

            INDArray oversampledFeaturesArray = Nd4j.vstack(oversampledFeatures);
            INDArray oversampledLabelsArray = Nd4j.vstack(oversampledLabels);

            DataSet oversampledDataSet = new DataSet(oversampledFeaturesArray, oversampledLabelsArray);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(oversampledDataSet);
            normalizer.transform(oversampledDataSet);

            return oversampledDataSet;
        }
        else {
            return null;
        }
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

    private MultiLayerNetwork createNetworkFromConfig(JSONObject config) {
        // Implement this method to initialize the network from a JSON config
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
