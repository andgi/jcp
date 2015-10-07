// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package se.hb.jcp.io;

import java.io.InputStream;
import java.io.IOException;

import cern.colt.matrix.DoubleMatrix1D;

import se.hb.jcp.cp.DataSet;

/**
 * Public abstract base class for data set readers.
 *
 * @author anders.gidenstam(at)hb.se
 */

public abstract class DataSetReader
{
    public abstract DataSet read(InputStream    source,
                                 DoubleMatrix1D template)
        throws IOException;

    public DataSet read(InputStream source)
        throws IOException
    {
        return read(source, null);
    }
}