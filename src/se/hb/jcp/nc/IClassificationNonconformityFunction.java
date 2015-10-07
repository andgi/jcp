// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package se.hb.jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

/**
 * Represents an instance of a specific non-conformity function for
 * conformal classification.
 *
 * Contract for JCP use:
 * 1. The non-conformity function must be serializable, both as untrained and
 *    as trained.
 */
public interface IClassificationNonconformityFunction
    extends java.io.Serializable
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

    public se.hb.jcp.ml.IClassifier getClassifier();
}
