package se.hb.jcp.nc;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.AbstractMatrix2D;

public interface IRegressionNonconformityFunction {
    public DoubleMatrix2D predict(AbstractMatrix2D x, double significance);
}
