// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import java.util.Map;
import java.util.TreeMap;

public class AverageClassificationNonconformityFunction
    implements IClassificationNonconformityFunction, java.io.Serializable
{
    int[] _class_count;
    int _n_classes;
    double[] _classes;
    Map<Double, Integer> _class_index = new TreeMap<Double, Integer>();
    float _n_instances;
    
    public AverageClassificationNonconformityFunction(double[] classes)
    {
        _n_classes = classes.length;
        _class_count = new int[_n_classes];
        _classes = classes;
        
        for (int i = 0; i < _n_classes; i++) {
            _class_index.put(_classes[i], i);
        }
    }

    public void fit(DoubleMatrix2D x, double[] y)
    {
        for (double y_ : y)
            _class_count[_class_index.get(y_)]++;
        _n_instances = y.length;
    }

    
    public void fit(DoubleMatrix2D xtr, double[] ytr,
                    DoubleMatrix1D xtest, double ytest)
    {
        for (double y_ : ytr)
            _class_count[_class_index.get(y_)]++;
        
        _class_count[_class_index.get(ytest)]++;
        _n_instances = ytr.length + 1;
    }

    @Deprecated
    public double[] calc_nc(DoubleMatrix2D x, double[] y)
    {
        double[] nc = new double[y.length];
        for (int i = 0; i < nc.length; i++) {
            nc[i] = 1 - (_class_count[_class_index.get(y[i])] / _n_instances);
        }
        return nc;
    }

    
    public double[] calc_nc(DoubleMatrix2D xtr, double[] ytr,
                            DoubleMatrix1D xtest, double ytest)
    {
        double[] nc = new double[ytr.length + 1];
        for (int i = 0; i < nc.length - 1; i++) {
            nc[i] = 1 - (_class_count[_class_index.get(ytr[i])] / _n_instances);
        }
        nc[nc.length - 1] = 1 - (_class_count[_class_index.get(ytest)] / _n_instances);
        
        return nc;
    }

    public double calculateNonConformityScore(DoubleMatrix1D x, double y)
    {
        double nc = 1 - (_class_count[_class_index.get(y)] / _n_instances);
        return nc;
    }

    public jcp.ml.IClassifier getClassifier()
    {
        return null;
    }
}
