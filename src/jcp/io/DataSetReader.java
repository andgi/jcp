// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.io;

import java.io.InputStream;
import java.io.IOException;

import cern.colt.matrix.DoubleMatrix2D;

import jcp.cp.DataSet;

/**
 * Public abstract base class for data set readers.
 *
 * @author anders.gidenstam(at)hb.se
 */

public abstract class DataSetReader
{
    public abstract DataSet read(InputStream    source,
                                 DoubleMatrix2D template)
        throws IOException;

    public DataSet read(InputStream source)
        throws IOException
    {
        return read(source, null);
    }
}
