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

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassifier;

public class NNClassifier extends ClassifierBase{
    
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
    public IClassifier fitNew(DoubleMatrix2D x, double[] y) {
        NNClassifier clone = new NNClassifier(_conf); 
        // TOFIX it only fit with Dataset object ! 
        //clone.fit(x);
        return clone;
    }

    @Override
    public double predict(DoubleMatrix1D instance) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'predict'");
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'nativeStorageTemplate'");
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'internalFit'");
    }
}
