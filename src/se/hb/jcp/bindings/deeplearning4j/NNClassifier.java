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

import se.hb.jcp.ml.ClassifierBase;
import se.hb.jcp.ml.IClassifier;

public class NNClassifier {
    public void test() {
        int numInput = 784; // Taille des données MNIST (28x28)
        int numHidden = 1000; // Nombre de neurones dans la couche cachée
        int numOutput = 10; // Nombre de classes de sortie
        double learningRate = 0.1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(numHidden)
                .activation(Activation.RELU).build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(numHidden).nOut(numOutput).activation(Activation.SOFTMAX).build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
    }
}
