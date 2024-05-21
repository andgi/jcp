package se.hb.jcp.cp;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public interface IConformalPredictor {
    void fit(DoubleMatrix2D xtr, double[] ytr, DoubleMatrix2D xcal, double[] ycal);
    double[] predictIntervals(DoubleMatrix1D x, double confidence);
    double[][] predictIntervals(DoubleMatrix2D x, double confidence);
    boolean isTrained();
    int getAttributeCount();
    DoubleMatrix1D nativeStorageTemplate();
}
