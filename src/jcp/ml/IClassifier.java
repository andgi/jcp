// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.ml;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public interface IClassifier
{
    public void fit(DoubleMatrix2D x, double[] y);
    public IClassifier fitNew(DoubleMatrix2D x, double[] y);

    public double predict(DoubleMatrix1D instance);
    public double predict(DoubleMatrix1D instance,
                          double[] probabilityEstimates);

    public int getAttributeCount();

    public DoubleMatrix1D nativeStorageTemplate();
}
