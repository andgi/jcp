// JCP - Java Conformal Prediction framework
// Copyright (C) 2014  Henrik Linusson
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
package se.hb.jcp.test;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.ObjectMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import java.util.Date;
import java.util.Random;
import se.hb.jcp.cp.InductiveConformalClassifier;
import se.hb.jcp.nc.*;
import se.hb.jcp.bindings.jlibsvm.SparseDoubleMatrix2D;

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
            new se.hb.jcp.bindings.jlibsvm.SparseDoubleMatrix2D(ytr.length, 1);
        DoubleMatrix2D xcal =
            //new DenseDoubleMatrix2D(ycal.length, 1);
            new se.hb.jcp.bindings.jlibsvm.SparseDoubleMatrix2D(ycal.length, 1);
        DoubleMatrix2D xtest =
            //new DenseDoubleMatrix2D(ytest.length, 1);
            new se.hb.jcp.bindings.jlibsvm.SparseDoubleMatrix2D(ytest.length, 1);
           
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
                new ClassProbabilityNonconformityFunction(new double[]{0.0, 1.0, 2.0});
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
