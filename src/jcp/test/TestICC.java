package jcp.test;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import java.util.Date;
import java.util.Random;
import jcp.cp.InductiveConformalClassifier;
import jcp.nc.*;
import jcp.bindings.libsvm.SparseDoubleMatrix2D;

public class TestICC {
    public TestICC() {
        super();
    }
    
    public static void main(String[] args){
        Random rand = new Random(new Date().getTime());
        
        double n = 0;
        
        double[] ytr = new double[100];
        double[] ycal = new double[299];
        double[] ytest = new double[100];
        
        DoubleMatrix2D xtr =
            //new DenseDoubleMatrix2D(ytr.length, 1);
            new jcp.bindings.libsvm.SparseDoubleMatrix2D(ytr.length, 1);
        DoubleMatrix2D xcal =
            //new DenseDoubleMatrix2D(ycal.length, 1);
            new jcp.bindings.libsvm.SparseDoubleMatrix2D(ycal.length, 1);
        DoubleMatrix2D xtest =
            //new DenseDoubleMatrix2D(ytest.length, 1);
            new jcp.bindings.libsvm.SparseDoubleMatrix2D(ytest.length, 1);
           
        for (int k = 0; k < 1000; k++) {
    
            for (int i = 0; i < ytr.length; i++) {
                double r = rand.nextDouble();
                if (r < 0.10)
                    ytr[i] = 0.0;
                else if (r < 0.50)
                    ytr[i] = 1.0;
                else
                    ytr[i] = 2.0;
            }
            
            for (int i = 0; i < ycal.length; i++) {
                double r = rand.nextDouble();
                if (r < 0.10)
                    ycal[i] = 0.0;
                else if (r < 0.50)
                    ycal[i] = 1.0;
                else
                    ycal[i] = 2.0;
            }
            
            for (int i = 0; i < ytest.length; i++) {
                double r = rand.nextDouble();
                if (r < 0.10)
                    ytest[i] = 0.0;
                else if (r < 0.50)
                    ytest[i] = 1.0;
                else
                    ytest[i] = 2.0;
            }
            
            InductiveConformalClassifier icc = new InductiveConformalClassifier(new double[]{0, 1, 2});
            icc._nc =
                //new AverageClassificationNonconformityFunction(new double[]{0, 1, 2});
                new SVMClassificationNonconformityFunction(new double[]{0.0, 1.0, 2.0});
            icc.fit(xtr, ytr, xcal, ycal);


            ObjectMatrix2D pred = null;
            try {
                pred = icc.predict(xtest, 0.05);
            } catch (Exception e){ e.printStackTrace(); }
            //System.out.println(pred);
            
    
            for (int i = 0; i < pred.rows(); i++){
                if ((Boolean)pred.get(i, (int)ytest[i])) {
                    n += 1;
                }
            }
        }
        
        System.out.println((n / ytest.length) / 1000);
    }
}