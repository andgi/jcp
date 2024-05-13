package se.hb.jcp.bindings.neuroph;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassProbabilityClassifier;
import se.hb.jcp.ml.IClassifier;

import java.io.File;
import java.util.List;
import java.util.ArrayList;

public class NNClassifier extends ClassifierBase implements IClassProbabilityClassifier {
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected NeuralNetwork _network;
    
    public NNClassifier() {
    }

    public NNClassifier(JSONObject configuration) {
        this();
        //we can think about a json file with the number of features, then hidden layers and output 
        //and transfert fonction ? 
        // and read weights ? 
        //_network = createNetworkFromConfig(configuration);
        //here we have to determine parameters (create new class for that ?)
    }


    public NNClassifier(String modelFilePath) {
        this();
        _network = NeuralNetwork.createFromFile(modelFilePath);
    
    }
    public NNClassifier(NeuralNetwork network) {
        _network = network;
    }

    
    public IClassifier fitNew(DoubleMatrix2D x, double[] y) {
        NeuralNetwork newNetwork = createAndTrainNetwork(x, y);
        return new NNClassifier(newNetwork);
    }

    
    public double predict(DoubleMatrix1D instance) {

        _network.setInput(instance.toArray());
        _network.calculate();
        return _network.getOutput()[0];
    }

    public double predict(DoubleMatrix1D instance, double[] probabilityEstimates) {
      
        _network.setInput(instance.toArray());
        
        _network.calculate();
    
        double[] output = _network.getOutput();
        
        System.arraycopy(output, 0, probabilityEstimates, 0, output.length);
        return output[0];
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        return _storageTemplate;
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) {
        _network = createAndTrainNetwork(x, y);
    }

    private NeuralNetwork createAndTrainNetwork(DoubleMatrix2D x, double[] y) {

        //NeuralNetwork neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.TANH, x.columns(), 3, 1);

        NeuralNetwork neuralNetwork = new MultiLayerPerceptron(x.columns(), 16, 1);
        
        DataSet dataSet = createDataSet(x, y);
        
        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setLearningRate(0.01);
        backPropagation.setMaxIterations(500);

        neuralNetwork.setLearningRule(backPropagation);
        
        neuralNetwork.learn(dataSet);

        return neuralNetwork;
    }
    private DataSet createDataSet(DoubleMatrix2D x, double[] y) {
     
        DataSet dataSet = new DataSet(x.columns(), 1); 
        for (int i = 0; i < x.rows(); i++) {
            double[] features = x.viewRow(i).toArray();
            double[] label = {y[i]};
            dataSet.add(features, label );
        }
        return dataSet;
    }

    private NeuralNetwork createNetworkFromConfig(JSONObject config) {
        /*List<Integer> neuronsInLayers = new ArrayList<>();
        if (config == null) {
            //default config
        }
        else {
            if (config.has("networkType")) {
                System.out.println("NETWORK");
                
            }
            if (config.has("num_inputs")) {
                int numInputs = config.getInt("num_inputs");
                neuronsInLayers.add(numInputs);

            }
            if (config.has("hidden_layers")) {
                int hiddenLayers = config.getInt("hidden_layers");
                neuronsInLayers.add(hiddenLayers);
            }
            if (config.has("num_outputs")) {
                int numOutputs = config.getInt("num_outputs");
                neuronsInLayers.add(numOutputs);
            }

        }*/

    
       
        
        

        //NeuralNetwork neuralNetwork = new MultiLayerPerceptron(neuronsInLayers, TransferFunctionType.TANH);
        //train ? 

        //return neuralNetwork;
        //return null;
        List<Integer> layerParam = new ArrayList<>();
        layerParam.add(4);
        layerParam.add(16);
        layerParam.add(1);
        NeuralNetwork neuralNetwork = new MultiLayerPerceptron(layerParam, TransferFunctionType.TANH);
        return neuralNetwork;
    }
}
