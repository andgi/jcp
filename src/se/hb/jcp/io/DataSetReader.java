// JCP - Java Conformal Prediction framework
// Copyright (C) 2014  Anders Gidenstam
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
