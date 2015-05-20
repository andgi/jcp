// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public interface IClassificationNonconformityFunction
{
    public void fit(DoubleMatrix2D x, double[] y);
    public IClassificationNonconformityFunction fitNew(DoubleMatrix2D x,
                                                       double[] y);

    // FIXME: This method is inefficient if the underlying classifier
    //        still needs everything in one X and Y structure.
    public IClassificationNonconformityFunction
        fitNew(DoubleMatrix2D xtr, double[] ytr,
               DoubleMatrix1D xtest, double ytest);

    @Deprecated
    public double[] calc_nc(DoubleMatrix2D x, double[] y);
    public double[] calc_nc(DoubleMatrix2D xtrain, double[] ytrain,
                            DoubleMatrix1D xtest, double ytest);

    public double calculateNonConformityScore(DoubleMatrix1D x, double y);

    public jcp.ml.IClassifier getClassifier();
}
