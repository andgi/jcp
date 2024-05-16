package se.hb.jcp.bindings.deeplearning4j;


import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Array;
import java.io.File;
import java.io.IOException;

import java.util.Arrays;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassProbabilityClassifier;
import se.hb.jcp.ml.IClassifier;

import java.io.File;
import java.util.List;
import java.util.ArrayList;

public class NN4jClassifier extends ClassifierBase implements IClassProbabilityClassifier, java.io.Serializable{
    private static final SparseDoubleMatrix1D _storageTemplate =
        new SparseDoubleMatrix1D(0);
    protected MultiLayerNetwork  _model;
    
    public NN4jClassifier() {
    }

    public NN4jClassifier(JSONObject configuration) {
        this();
        


    }


    public NN4jClassifier(String modelFilePath) {
        this();
       
    
    }
    public NN4jClassifier(MultiLayerNetwork  model) {
        _model = model;
    }

    
    public IClassifier fitNew(DoubleMatrix2D x, double[] y) {
        internalFit(x, y);
        return new NN4jClassifier(_model);
    }

    
    public double predict(DoubleMatrix1D instance) {

        INDArray input = Nd4j.create(instance.toArray());
        INDArray output = _model.output(input);
        return output.getDouble(0);
    }

    public double predict(DoubleMatrix1D instance, double[] probabilityEstimates) {
      
        INDArray input = Nd4j.create(instance.toArray()).reshape(1, instance.size());        
        INDArray output = _model.output(input);
        double[] probabilities = output.toDoubleVector();
        System.arraycopy(probabilities, 0, probabilityEstimates, 0, probabilities.length);
        
        return output.getDouble(0);
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        return _storageTemplate;
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) {
        _model = createAndTrainNetwork(x, y);
    }

    private MultiLayerNetwork  createAndTrainNetwork(DoubleMatrix2D x, double[] y) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(x.columns()).nOut(3)
                        .build())
                .layer(new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(3).nOut(1).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

    
        DataSet dataSet = createDataSet(x, y);

    
        model.fit(dataSet);
        //System.out.println(dataSet);

        return model;
    }
    
    private DataSet createDataSet(DoubleMatrix2D x, double[] y) {
        
        int rows = x.rows();
        int cols = x.columns();
        //System.out.println(x);
        /*double[][] array = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                array[i][j] = x.get(i, j);
                if (x.get(i,j) != 0) {
                    System.out.println(x.get(i,j));
                }
            }
        }*/
        //issue here
        //System.out.println(Arrays.toString(x.toArray()[1]));
        

        INDArray features = Nd4j.create(x.toArray());
        System.out.println(features);
        //System.out.println(features);
        
        INDArray labels = Nd4j.create(y, new int[]{rows, 1});
  

        DataSet dataSet = new DataSet(features, labels);
        
        // Normalize the data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);

        return dataSet;
    }

    private MultiLayerNetwork createNetworkFromConfig(JSONObject config) {
      return null;
    }
   

    private void writeObject(ObjectOutputStream oos) throws IOException {
        if (_model != null) {
            // Create a (likely) unique file name for the Java libsvm model.
            String fileName =
                Long.toHexString(Double.doubleToLongBits(Math.random())) +
                ".deeplearning4j";

            // Save the model to a separate file.
            _model.save(new File(fileName));
            // Save the model file name.
            oos.writeObject(fileName);
        } else {
            // Save null if the model has not been trained.
            oos.writeObject(null);
        }
    }
    
    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
   
            String fileName = (String)ois.readObject();
            if (fileName != null) {
                // Load the model from the designated file.
                _model = MultiLayerNetwork.load(new File(fileName), true);
            }
    }
}
