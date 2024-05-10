package se.hb.jcp.bindings.deeplearning4j;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassProbabilityClassifier;
import se.hb.jcp.ml.IClassifier;

public class NNClassifier extends ClassifierBase implements IClassProbabilityClassifier {
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected NeuralNetwork _network;

    public NNClassifier() {
    }

    public NNClassifier(JSONObject configuration) {
        this();
        //we can think about a json file with the number of features, then hidden layers and output 
    }

    public NNClassifier(NeuralNetwork network) {
        _network = network;
    }

    @Override
    public IClassifier fitNew(DoubleMatrix2D x, double[] y) {
        NeuralNetwork newNetwork = createAndTrainNetwork(x, y);
        return new NNClassifier(newNetwork);
    }

    @Override
    public double predict(DoubleMatrix1D instance) {
        return 0.0;
    }

    public double predict(DoubleMatrix1D instance, double[] probabilityEstimates) {
        return 0.0;
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        return _storageTemplate;
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) {
        // Implement training
    }

    private NeuralNetwork createAndTrainNetwork(DoubleMatrix2D x, double[] y) {
       
        NeuralNetwork neuralNetwork = new MultiLayerPerceptron(10, 20, 1);

        DataSet dataSet = createDataSet(x, y);
        
        neuralNetwork.learn(dataSet);
        return neuralNetwork;
    }
    private DataSet createDataSet(DoubleMatrix2D x, double[] y) {
        DataSet dataSet = new DataSet(1000, y.length);

        for (int i = 0; i < x.rows(); i++) {
            double[] features = x.viewRow(i).toArray();
            double label = y[i];
            DataSetRow dataSetRow = new DataSetRow(features, new double[]{label});
            //TOFIX
            //dataSet.add(dataSetRow);
        }
        return dataSet;
    }
}
