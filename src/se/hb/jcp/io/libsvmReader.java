// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2016, 2019  Anders Gidenstam
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
package se.hb.jcp.io;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.ArrayList;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleFactory2D;

import se.hb.jcp.cp.DataSet;

/**
 * Data set reader for the libsvm sparse data format.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class libsvmReader
    extends DataSetReader
{
    @Override
    public DataSet read(InputStream    source,
                        DoubleMatrix1D template)
        throws IOException
    {
        DataSet p;
        // Read through the entire data set file and collect the information
        // needed to instantiate the x and y matrices.
        // FIXME: This is not very memory efficient.
        try (BufferedReader br = new BufferedReader(new InputStreamReader(source))) {
            // Read through the entire data set file and collect the information
            // needed to instantiate the x and y matrices.
            // FIXME: This is not very memory efficient.
            int rows = 0;
            int columns = 0;
            ArrayList<Double>   vy = new ArrayList<Double>();
            ArrayList<int[]>    vxi = new ArrayList<int[]>();
            ArrayList<double[]> vxv = new ArrayList<double[]>();
            while(true) {
                String line = br.readLine();
                if(line == null) break;

                StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

                vy.add(parseDouble(st.nextToken()));
                int m = st.countTokens()/2;
                int[]    xi = new int[m];
                double[] xv = new double[m];
                for (int j=0; j<m; j++) {
                    xi[j] = parseInt(st.nextToken());
                    xv[j] = parseDouble(st.nextToken());
                }
                if (m > 0) {
                    columns = Math.max(columns, xi[m-1]);
                }
                vxi.add(xi);
                vxv.add(xv);
                rows++;
            }
            p = new DataSet();
            // Create and initialize y.
            p.y = new double[rows];
            //p.y = DoubleFactory1D.dense.make(rows);
            for (int r = 0; r < p.y.length; r++) {
                p.y[r] = vy.get(r);
                //p.y.set(r, vy.get(r));
            }
            // Create and initialize x.
            if (template != null) {
                p.x = template.like2D(rows, columns);
            } else {
                // Default to libsvm data storage.
                //p.x = new se.hb.jcp.bindings.libsvm.SparseDoubleMatrix2D(rows, columns);
                // Default to colt data storage.
                p.x = DoubleFactory2D.sparse.make(rows, columns);
            }
            for (int r = 0; r < p.x.rows(); r++) {
                int[]    xi = vxi.get(r);
                double[] xv = vxv.get(r);
                for (int a = 0; a < xi.length; a++) {
                    p.x.set(r, xi[a] - 1, xv[a]);
                }
            }
        }

        return p;
    }

    private static double parseDouble(String s)
        throws IOException
    {
        double d = Double.parseDouble(s);
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            throw new IOException("NaN or Infinity in input data file.");
        }
        return d;
    }

    private static int parseInt(String s)
    {
        return Integer.parseInt(s);
    }
}
