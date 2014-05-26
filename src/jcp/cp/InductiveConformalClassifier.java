package jcp.cp;

import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseObjectMatrix2D;

import java.util.Arrays;

import jcp.nc.IClassificationNonconformityFunction;

public class InductiveConformalClassifier { 
    public IClassificationNonconformityFunction _nc;
    private double[] _calibration_scores;
    private double[] _targets;
    
    private DoubleMatrix2D _xtr;
    private DoubleMatrix2D _xcal;
    
    private double[] _ytr;
    private double[] _ycal;

    public InductiveConformalClassifier(double[] targets) {
        _targets = targets;
    }

    public void fit(DoubleMatrix2D xtr, double[] ytr,
                    DoubleMatrix2D xcal, double[] ycal) {
        _xtr = xtr;
        _ytr = ytr;
        _xcal = xcal;
        _ycal = ycal;
        
        _nc.fit(_xtr, _ytr);
        _calibration_scores = _nc.calc_nc(_xcal, _ycal);
        Arrays.sort(_calibration_scores);
    }

    public ObjectMatrix2D predict(DoubleMatrix2D x, double significance) {
        int n = x.rows();
        ObjectMatrix2D response = new SparseObjectMatrix2D(n, _targets.length);
        double[] y = new double[n];
        
        for (int i = 0; i < _targets.length; i++) {
            for (int j = 0; j < n; j++)
                y[j] = _targets[i];
            // TODO: Underlying model should really only have to predict once.
            double[] nc_pred = _nc.calc_nc(x, y);
            boolean[] include = Util.calc_inclusion(nc_pred, _calibration_scores, significance);
            for (int j = 0; j < n; j++) {
                response.set(j, i, include[j]);
            }
        }
        return response;
    }
}
