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
package se.hb.jcp.bindings.neuroph;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.File;
import java.io.IOException;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.core.Neuron;
import org.neuroph.core.Layer;
import org.neuroph.core.transfer.Tanh;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.supervised.instance.SMOTE;
import se.hb.jcp.bindings.deeplearning4j.WekaUtils;




import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassProbabilityClassifier;
import se.hb.jcp.ml.IClassifier;

import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;


import java.io.File;
import java.util.List;
import java.util.ArrayList;

public class NNClassifier extends ClassifierBase implements IClassProbabilityClassifier, LearningEventListener, java.io.Serializable{
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

        double probability = output[0];
        probabilityEstimates[0] = 1 - probability;
        probabilityEstimates[1] = probability;
        System.out.println(Arrays.toString(output));
        return (probability >= 0.5) ? 1.0 : -1.0;
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

        NeuralNetwork neuralNetwork = new MultiLayerPerceptron(x.columns(), 10, 1);
        /*TransferFunction softmax = new SoftMax(neuralNetwork.getLayerAt(neuralNetwork.getLayersCount() - 1));
        Neuron last = (Neuron) neuralNetwork.getOutputNeurons().get(0);
        last.setTransferFunction(softmax);*/
        /*Layer lastLayer = neuralNetwork.getLayerAt(neuralNetwork.getLayersCount() - 1);

        Neuron last = lastLayer.getNeuronAt(lastLayer.getNeuronsCount() - 1);
        last.setTransferFunction(new Tanh());*/
        //DoubleMatrix2D normalizedX = normalizeData(x);
        DataSet dataSet = null;
        try {
            dataSet = dataSet = createBalancedDataSet(x, y);
        } catch (Exception e) {
            e.printStackTrace();
        }

        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNetwork.getLearningRule();
        learningRule.addListener(this);

        learningRule.setLearningRate(0.2);
        learningRule.setMaxError(0.05);
        learningRule.setMaxIterations(200);
        neuralNetwork.learn(dataSet);

        return neuralNetwork;
    }
    private DataSet createBalancedDataSet(DoubleMatrix2D x, double[] y) throws Exception {
        Instances wekaInstances = convertToWekaInstances(x, y);

        SMOTE smote = new SMOTE();
        smote.setInputFormat(wekaInstances);
        smote.setPercentage(100.0);

        Instances balancedWekaInstances = Filter.useFilter(wekaInstances, smote);

        DataSet balancedDataSet = new DataSet(x.columns(), 1);

        for (int i = 0; i < balancedWekaInstances.size(); i++) {
            double[] features = new double[x.columns()];
            double[] labels = new double[1];

            for (int j = 0; j < x.columns(); j++) {
                features[j] = balancedWekaInstances.instance(i).value(j);
            }
            labels[0] = balancedWekaInstances.instance(i).classValue() == 0.0 ? -1.0 : 1.0;

            balancedDataSet.add(new DataSetRow(features, labels));
        }

        Normalizer normalizer = new MaxNormalizer(balancedDataSet);
        normalizer.normalize(balancedDataSet);

        return balancedDataSet;
    }


    private Instances convertToWekaInstances(DoubleMatrix2D x, double[] y) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < x.columns(); i++) {
            attributes.add(new Attribute("attr" + i));
        }
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("0"); // for -1.0
        classValues.add("1"); // for 1.0
        attributes.add(new Attribute("class", classValues));

        Instances instances = new Instances("dataset", attributes, x.rows());
        instances.setClassIndex(instances.numAttributes() - 1);

        for (int i = 0; i < x.rows(); i++) {
            double[] instanceValues = new double[instances.numAttributes()];
            for (int j = 0; j < x.columns(); j++) {
                instanceValues[j] = x.get(i, j);
            }
            instanceValues[instances.numAttributes() - 1] = (y[i] == -1.0) ? 0.0 : 1.0;
            instances.add(new DenseInstance(1.0, instanceValues));
        }

        return instances;
    }

    /*private DataSet createDataSet(DoubleMatrix2D x, double[] y) {
        DataSet dataSet = new DataSet(x.columns(), 1);
        int minorClassCount = 0;
        for (double label : y) {
            if (label == -1.0) {
                minorClassCount++;
            }
        }

        int[] minorClassIndices = new int[minorClassCount];
        int index = 0;
        for (int i = 0; i < y.length; i++) {
            if (y[i] == -1.0) {
                minorClassIndices[index++] = i;
            }
        }

        for (int i = 0; i < x.rows(); i++) {
            double[] features = x.viewRow(i).toArray();
            double[] label = {y[i]};
            dataSet.add(new DataSetRow(features, label));
        }

        while (minorClassCount < y.length / 4) {
            int randomIndex = minorClassIndices[(int) (Math.random() * minorClassIndices.length)];
            double[] features = x.viewRow(randomIndex).toArray();
            double[] label = {y[randomIndex]};
            dataSet.add(new DataSetRow(features, label));
            minorClassCount++;
        }
        return dataSet;
    }*/

    /*private DataSet createDataSet(DoubleMatrix2D x, double[] y) {
        DataSet dataSet = new DataSet(x.columns(), 1); 
        for (int i = 0; i < x.rows(); i++) {
            double[] features = x.viewRow(i).toArray();
            double[] label = {y[i]};
            dataSet.add(features, label);
        }
        return dataSet;
    }*/

    private DoubleMatrix2D normalizeData(DoubleMatrix2D data) {
        for (int i = 0; i < data.columns(); i++) {
            double mean = 0;
            double std = 0;
            for (int j = 0; j < data.rows(); j++) {
                mean += data.get(j, i);
            }
            mean /= data.rows();
            for (int j = 0; j < data.rows(); j++) {
                std += Math.pow(data.get(j, i) - mean, 2);
            }
            std = Math.sqrt(std / data.rows());
            for (int j = 0; j < data.rows(); j++) {
                data.set(j, i, (data.get(j, i) - mean) / std);
            }
        }
        return data;
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

    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation) event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.defaultWriteObject(); // Write non-transient fields

        // Save the model if it has been trained
        if (_network != null) {
            // Save the neural network to a file

            String modelFileName =
            Long.toHexString(Double.doubleToLongBits(Math.random())) +
                ".neuroph";
            _network.save(modelFileName);
            // Write the model file name to the stream
            oos.writeObject(modelFileName);
        } else {
            // Indicate that the model has not been trained
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        ois.defaultReadObject(); // Read non-transient fields

        // Load model file name from the stream
        String modelFileName = (String) ois.readObject();
        if (modelFileName != null) {
            File file = new File(modelFileName);
            // Load the neural network from the saved file
            _network = NeuralNetwork.createFromFile(file);
        }
    }
}
