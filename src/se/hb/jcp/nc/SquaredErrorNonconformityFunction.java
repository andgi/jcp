package se.hb.jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.AbstractMatrix2D;
import se.hb.jcp.ml.IRegressor;

public class SquaredErrorNonconformityFunction implements IRegressionNonconformityFunction,  java.io.Serializable{
    private IRegressor _regressor;
    private boolean _isTrained = false;

    public SquaredErrorNonconformityFunction(IRegressor regressor) {
        _regressor = regressor;
    }

    @Override
    public void fit(DoubleMatrix2D x, double[] y) {
        _regressor.fit(x, y);
        _isTrained = true;
    }

    @Override
    public boolean isTrained() {
        return _isTrained;
    }

    @Override
    public double calculateNonConformityScore(DoubleMatrix1D instance, double label) {
        double prediction = _regressor.predict(instance);
        double error = label - prediction;
        return error * error;
    }

    @Override
    public double predict(DoubleMatrix1D instance) {
        return _regressor.predict(instance);
    }

    @Override
    public int getAttributeCount() {
        return _regressor.getAttributeCount();
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        return _regressor.nativeStorageTemplate();
    }

    @Override
    public DoubleMatrix2D predict(AbstractMatrix2D x, double significance) {
        //TODO
        System.out.println("To implement");
        return null; 
    }
}
