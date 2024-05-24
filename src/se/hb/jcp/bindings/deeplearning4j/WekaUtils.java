package se.hb.jcp.bindings.deeplearning4j;

import org.nd4j.linalg.api.ndarray.INDArray;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;

public class WekaUtils {
    public static Instances convertToWekaInstances(INDArray features, INDArray labels) {
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
}

