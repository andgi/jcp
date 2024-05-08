package se.hb.jcp.bindings.deeplearning4j;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.DataSet; 
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassProbabilityClassifier;
import se.hb.jcp.ml.IClassifier;

public class NNClassifier extends ClassifierBase implements IClassProbabilityClassifier{
    
    protected MultiLayerConfiguration _conf; 
    protected MultiLayerNetwork _model; 

    public NNClassifier() {
    }
    public NNClassifier(JSONObject configuration) {
        this();
        //TOFIX I dont know if its work like this ? 
        _conf = MultiLayerConfiguration.fromJson(configuration.toString());
    }

    public NNClassifier(MultiLayerConfiguration configuration) {
        _conf = configuration;
        _model = new MultiLayerNetwork(_conf);
        _model.init();
    }

    @Override
    public IClassifier fitNew(DoubleMatrix2D x, double[] y) 
    {
        NNClassifier clone = new NNClassifier(_conf); 
        clone.fit(x, y);
        return clone;
    }

    @Override
    public double predict(DoubleMatrix1D instance) 
    {
        INDArray input = Nd4j.create(instance.toArray());
        INDArray output = _model.output(input);
        double prediction = output.getDouble(0); 
        return prediction;
    }
     public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates)
    {
        INDArray input = Nd4j.create(instance.toArray());
        INDArray output = _model.output(input);
        double prediction = output.getDouble(0); 
        for (int i = 0; i < output.length(); i++) {
            probabilityEstimates[i] = output.getDouble(i);
        }
        
        return prediction;

    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'nativeStorageTemplate'");
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) 
    {
        INDArray features = Nd4j.create(x.toArray());
        INDArray labels = Nd4j.create(y, new int[]{y.length, 1});
        DataSet dataSet = new DataSet(features, labels);
        MultiLayerNetwork model = new MultiLayerNetwork(_conf);
        model.init();
        model.fit(dataSet);
        _model = model;
    }
}
