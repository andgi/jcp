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
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;

import org.json.JSONObject;

import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.evaluation.classification.Evaluation;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.ClassBalancer;

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
        //_model = createNetworkFromConfig(configuration);
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

        probabilityEstimates[0] = output.getDouble(0);
        probabilityEstimates[1] = output.getDouble(1);

        System.out.println(Arrays.toString(probabilityEstimates));
        return output.getDouble(0) >= output.getDouble(1) ? -1.0 : 1.0;
    }

    public double findBestThreshold(DataSet validationData) {
        double bestThreshold = 0.0;
        double bestF1Score = 0.0;

        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.05) {
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
        DataSet dataSet = null;
        try {
            dataSet = createDataSetWithClassBalancer(x, y);
        } catch (Exception e) {
            e.printStackTrace();
        }

        int seed = 123;
        double learningRate = 0.005;
        int nEpochs = 100;

        int numInputs = x.columns();
        int numOutputs = 2;
        int numHiddenNodes = 50;

        DataSetIterator dataIter = new ViewIterator(dataSet, 100);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(dataIter, nEpochs);

        System.out.println("Evaluate model....");
        Evaluation eval = model.evaluate(dataIter);

        System.out.println(eval.stats());
        System.out.println("\n****************Example finished********************");
        _model = model;
        return model;
    }

    private DataSet createDataSetWithClassBalancer(DoubleMatrix2D x, double[] y) throws Exception {
        INDArray features = Nd4j.create(x.toArray());
        INDArray labels = Nd4j.create(y, new long[]{y.length, 1});

        Instances wekaInstances = convertToWekaInstances(features, labels);

        ClassBalancer classBalancer = new ClassBalancer();
        classBalancer.setInputFormat(wekaInstances);

        Instances balancedWekaInstances = Filter.useFilter(wekaInstances, classBalancer);

        int numOutputs = 2;
        INDArray balancedFeatures = Nd4j.create(balancedWekaInstances.size(), x.columns());
        INDArray balancedLabels = Nd4j.create(balancedWekaInstances.size(), numOutputs);

        for (int i = 0; i < balancedWekaInstances.size(); i++) {
            for (int j = 0; j < x.columns(); j++) {
                balancedFeatures.putScalar(new int[]{i, j}, balancedWekaInstances.instance(i).value(j));
            }
            int classValue = (int) balancedWekaInstances.instance(i).classValue();
            balancedLabels.putScalar(new int[]{i, classValue}, 1.0);
        }

        DataSet balancedDataSet = new DataSet(balancedFeatures, balancedLabels);
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(balancedDataSet);
        normalizer.transform(balancedDataSet);

        return balancedDataSet;
    }

    private Instances convertToWekaInstances(INDArray features, INDArray labels) {
        ArrayList<Attribute> attributes = new ArrayList<>();

        for (int i = 0; i < features.columns(); i++) {
            attributes.add(new Attribute("feature" + i));
        }

        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("class0");
        classValues.add("class1");
        attributes.add(new Attribute("class", classValues));

        Instances dataset = new Instances("Dataset", attributes, features.rows());
        dataset.setClassIndex(dataset.numAttributes() - 1);

        for (int i = 0; i < features.rows(); i++) {
            double[] instanceValues = new double[features.columns() + 1];
            for (int j = 0; j < features.columns(); j++) {
                instanceValues[j] = features.getDouble(i, j);
            }
            instanceValues[features.columns()] = labels.getDouble(i) == -1.0 ? 0 : 1;
            dataset.add(new DenseInstance(1.0, instanceValues));
        }

        return dataset;
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
    
        if (_model != null) {

            oos.writeObject(_threshold);
            String fileName = Long.toHexString(Double.doubleToLongBits(Math.random())) + ".deeplearning4j";
            _model.save(new File(fileName));
            oos.writeObject(fileName);

        } else {
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
       
        double thresholdText = (double)ois.readObject();
        _threshold = (double) thresholdText; 
    
        String fileName = (String) ois.readObject();
        if (fileName != null) {
            _model = MultiLayerNetwork.load(new File(fileName), true);
        }
        
    }


    public double[] getClassProbabilities(DoubleMatrix1D instance) {
        INDArray input = Nd4j.create(instance.toArray()).reshape(1, instance.size());
        INDArray output = _model.output(input);
        return output.toDoubleVector();
    }

    private MultiLayerNetwork createNetworkFromConfig(JSONObject configuration) {

        int seed = configuration.getInt("seed");
        double learningRate = configuration.getDouble("learningRate");
        int numInputs = configuration.getInt("numInputs");
        int numOutputs = configuration.getInt("numOutputs");
        int numHiddenNodes = configuration.getInt("numHiddenNodes");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        return model;
    }
}
