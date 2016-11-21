// JCP - Java Conformal Prediction framework
// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015 - 2016  Anders Gidenstam
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
package se.hb.jcp.nc;

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
    int _n_instances;
    int _attributeCount = -1;

    public AverageClassificationNonconformityFunction(double[] classes)
    {
        _n_classes = classes.length;
        _class_count = new int[_n_classes];

        _classes = classes;
        for (int i = 0; i < _n_classes; i++) {
            _class_index.put(_classes[i], i);
        }
    }

    @Override
    public void fit(DoubleMatrix2D x, double[] y)
    {
        for (double y_ : y)
            _class_count[_class_index.get(y_)]++;
        _n_instances = y.length;
        _attributeCount = x.size();
    }

    
    private void fit(DoubleMatrix2D xtr, double[] ytr,
                     DoubleMatrix1D xtest, double ytest)
    {
        for (double y_ : ytr)
            _class_count[_class_index.get(y_)]++;
        
        _class_count[_class_index.get(ytest)]++;
        _n_instances = ytr.length + 1;
    }

    @Override
    public IClassificationNonconformityFunction fitNew(DoubleMatrix2D x,
                                                       double[] y)
    {
        // FIXME: Part of the work for the same training set could be shared.
        AverageClassificationNonconformityFunction nc =
            new AverageClassificationNonconformityFunction(_classes);
        nc.fit(x, y);
        return nc;
    }

    @Override
    public IClassificationNonconformityFunction
        fitNew(DoubleMatrix2D xtr, double[] ytr,
               DoubleMatrix1D xtest, double ytest)
    {
        // FIXME: Part of the work for the same training set could be shared.
        AverageClassificationNonconformityFunction nc =
            new AverageClassificationNonconformityFunction(_classes);
        nc.fit(xtr, ytr, xtest, ytest);
        return nc;
    }

    @Deprecated
    @Override
    public double[] calc_nc(DoubleMatrix2D x, double[] y)
    {
        double[] nc = new double[y.length];
        for (int i = 0; i < nc.length; i++) {
            nc[i] =
                1 - (double)_class_count[_class_index.get(y[i])] / _n_instances;
        }
        return nc;
    }

    @Override
    public double[] calc_nc(DoubleMatrix2D xtr, double[] ytr,
                            DoubleMatrix1D xtest, double ytest)
    {
        double[] nc = new double[ytr.length + 1];
        for (int i = 0; i < nc.length - 1; i++) {
            nc[i] = 1 - (double)_class_count[_class_index.get(ytr[i])] /
                        _n_instances;
        }
        nc[nc.length - 1] =
            1 - (double)_class_count[_class_index.get(ytest)] / _n_instances;
        
        return nc;
    }

    @Override
    public double calculateNonConformityScore(DoubleMatrix1D x, double y)
    {
        double nc = 1 - (double)_class_count[_class_index.get(y)] /
                        _n_instances;
        return nc;
    }

    @Override
    public se.hb.jcp.ml.IClassifier getClassifier()
    {
        return null;
    }

    @Override
    public final boolean isTrained()
    {
        return getAttributeCount() != -1;
    }

    @Override
    public final int getAttributeCount()
    {
        return _attributeCount;
    }

    @Override
    public final Double[] getLabels()
    {
        return _class_index.keySet().toArray(new Double[0]);
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate()
    {
        return new cern.colt.matrix.impl.SparseDoubleMatrix1D(0);
    }
}
