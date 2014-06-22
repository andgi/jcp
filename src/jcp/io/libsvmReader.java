// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.io;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Vector;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;

import jcp.cp.DataSet;

/**
 * Data set reader for the libsvm sparse data format.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class libsvmReader
    extends DataSetReader
{
    public DataSet read(InputStream    source,
                        DoubleMatrix2D template)
        throws IOException
    {
        BufferedReader br =
            new BufferedReader(new InputStreamReader(source));

        // Read through the entire data set file and collect the information
        // needed to instantiate the x and y matrices.
        // FIXME: This is not very memory efficient.
        int rows = 0;
        int columns = 0;
        Vector<Double>   vy = new Vector<Double>();
        Vector<int[]>    vxi = new Vector<int[]>();
        Vector<double[]> vxv = new Vector<double[]>();
        while(true) {
            String line = br.readLine();
            if(line == null) break;

            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

            vy.addElement(parseDouble(st.nextToken()));
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
            vxi.addElement(xi);
            vxv.addElement(xv);
            rows++;
        }

        DataSet p = new DataSet();
        // Create and initialize y.
        p.y = new double[rows];
        //p.y = DoubleFactory1D.dense.make(rows);
        for (int r = 0; r < p.y.length; r++) {
            p.y[r] = vy.get(r);
            //p.y.set(r, vy.get(r));
        }

        // Create and initialize x.
        if (template != null) {
            p.x = template.like(rows, columns);
        } else {
            // Default to libsvm data storage.
            //p.x = new jcp.bindings.libsvm.SparseDoubleMatrix2D(rows, columns);
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
        br.close();

        return p;
    }

    private static double parseDouble(String s)
        throws IOException
    {
        double d = Double.valueOf(s).doubleValue();
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
