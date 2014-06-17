// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.cp;

import java.util.Date;
import java.util.Random;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

/**
 * A data set.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class DataSet
{
    public DoubleMatrix2D x;
    //public DoubleMatrix1D y;
    public double[] y;

    public void random3Partition(DataSet p1, DataSet p2, DataSet p3,
                                 double f1, double f2)
    {
        int size1 = (int)(f1 * x.rows());
        int size2 = (int)(f2 * x.rows());
        int size3 = x.rows() - size1 - size2;
        Random random = new Random(new Date().getTime());

        p1.x = x.like(size1, x.columns());
        p1.y = new double[size1];
        p2.x = x.like(size2, x.columns());
        p2.y = new double[size2];
        p3.x = x.like(size3, x.columns());
        p3.y = new double[size3];

        // FIXME: Verify the uniformity of the partitioning scheme.
        int r1 = 0;
        int r2 = 0;
        int r3 = 0;
        for (int r = 0; r < x.rows(); r++) {
            double p = random.nextDouble();
            if (p < f1 && r1 < size1) {
                p1.copyInstance(r1, this, r);
                r1++;
            } else if (p < f1 + f2 && r2 < size2) {
                p2.copyInstance(r2, this, r);
                r2++;
            } else if (r3 < size3) {
                p3.copyInstance(r3, this, r);
                r3++;
            } else {
                r--;
            }
        }
    }

    private void copyInstance(int destinationRow, DataSet source, int sourceRow)
    {
        y[destinationRow] = source.y[sourceRow];
        // FIXME: Make this loop sparse!
        for (int a = 0; a < source.x.columns(); a++) {
            double v = source.x.get(sourceRow, a);
            if (v != 0.0) {
                x.set(destinationRow, a, v);
            }
        }
    }
}
