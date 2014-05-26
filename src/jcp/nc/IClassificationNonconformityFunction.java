package jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public interface IClassificationNonconformityFunction {
    public void fit(DoubleMatrix2D x, double[] y);
    public void fit(DoubleMatrix2D xtr, double[] ytr,
                    DoubleMatrix1D xtest, double ytest);
    
    public double[] calc_nc(DoubleMatrix2D x, double[] y);
    public double[] calc_nc(DoubleMatrix2D xtrain, double[] ytrain,
                            DoubleMatrix1D xtest, double ytest);
}
