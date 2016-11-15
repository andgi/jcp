// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2016  Anders Gidenstam
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
package se.hb.jcp.cp;

import java.util.Date;
import java.util.Random;

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

        p1.x = newFromTemplate(p1.x, size1, x.columns());
        p1.y = new double[size1];
        p2.x = newFromTemplate(p2.x, size2, x.columns());
        p2.y = new double[size2];
        p3.x = newFromTemplate(p3.x, size3, x.columns());
        p3.y = new double[size3];

        // FIXME: Verify the uniformity of the partitioning scheme.
        int r1 = 0;
        int r2 = 0;
        int r3 = 0;
        for (int r = 0; r < x.rows(); r++) {
            double p = random.nextDouble();
            if (p < f1 && r1 < size1) {
                p1.copyInstanceTo(r1, this, r);
                r1++;
            } else if (p < f1 + f2 && r2 < size2) {
                p2.copyInstanceTo(r2, this, r);
                r2++;
            } else if (r3 < size3) {
                p3.copyInstanceTo(r3, this, r);
                r3++;
            } else {
                r--;
            }
        }
    }

    private void copyInstanceTo(int destinationRow,
                                DataSet source, int sourceRow)
    {
        y[destinationRow] = source.y[sourceRow];
        x.viewRow(destinationRow).assign(source.x.viewRow(sourceRow));
    }

    private DoubleMatrix2D newFromTemplate(DoubleMatrix2D template,
                                           int rows,
                                           int columns)
    {
        if (template != null) {
            return template.like(rows, columns);
        } else {
            return x.like(rows, columns);
        }
    }
}
