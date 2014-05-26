package jcp.cp;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.impl.SparseObjectMatrix2D;

import java.util.Arrays;

import jcp.nc.IClassificationNonconformityFunction;

public class TransductiveConformalClassifier {
    public IClassificationNonconformityFunction _nc;
    private double[] _targets;
    
    private DoubleMatrix2D _xtr;   
    private double[] _ytr;

    public TransductiveConformalClassifier(double[] targets) {
        _targets = targets;
    }

    public void fit(DoubleMatrix2D xtr, double[] ytr) {
        _xtr = xtr;
        _ytr = ytr;
    }
    
    public ObjectMatrix2D predict(DoubleMatrix2D x, double significance) {
        int n = x.rows();
        ObjectMatrix2D response = new SparseObjectMatrix2D(n, _targets.length);
        
        for (int i = 0; i < _targets.length; i++) {
            for (int j = 0; j < n; j++) {
                boolean[] include = _predict(x.viewRow(j), _targets[i], significance);
                response.set(j, i, include[0]);
            }
        }
        return response;
    }

    private boolean[] _predict(DoubleMatrix1D xtest, double ytest, double significance) {
        _nc.fit(_xtr, _ytr, xtest, ytest);
        double[] nc = _nc.calc_nc(_xtr, _ytr, xtest, ytest);
        double[] nc_cal = Arrays.copyOf(nc, nc.length - 1);
        Arrays.sort(nc_cal);
        nc = new double[] { nc[nc.length - 1] };
        return Util.calc_inclusion(nc, nc_cal, significance);
    }
}
