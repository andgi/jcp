// JCP - Java Conformal Prediction framework
// Copyright (C) 2015, 2019  Anders Gidenstam
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
// The public interface is based on cern.colt.matrix.DoubleMatrix2D.
package se.hb.jcp.bindings.deeplearning4j;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;

/**
 * Class for sparse 2-d matrices holding <tt>double</tt> elements in
 * the sparse format expected by the Java library liblinear. See the
 * documentation for liblinear for more details.
 *
 * @author anders.gidenstam(at)hb.se
*/
// TODO: Make sure to adhere to colt's conventions.

public class SparseDoubleMatrix2D extends cern.colt.matrix.DoubleMatrix2D
{
    /**
     * Internal array of array of Feature nodes as the Java implementation of
     * liblinear expects.
     * NOTE: Internal Feature indices MUST start from 1.
     */
    Feature[][] _rows;

    /**
     * SparseDoubleMatrix1D views of rows in this matrix.
     * These are created and stored on demand.
     */
    protected SparseDoubleMatrix1D[] _rowViews;

    /**
     * Constructs a matrix with a given number of rows and columns.
     * All entries are initially <tt>0</tt>.
     * @param rows the number of rows the matrix shall have.
     * @param columns the number of columns the matrix shall have.
     * @throws IllegalArgumentException if
               <tt>rows&lt;0 || columns&lt;0 || (double)columns*rows &gt;
               Integer.MAX_VALUE</tt>.
    */
    public SparseDoubleMatrix2D(int rows, int columns)
    {
        setUp(rows, columns);
        for (int r = 0; r < rows; r++) {
            _rows[r] = new Feature[0];
        }
    }

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic type</i>
     * as the receiver, having the specified number of rows and columns.
     * For example, if the receiver is an instance of type
     * <tt>SparseDoubleMatrix2D</tt> the new matrix must also be of type
     * <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of type
     * <tt>SparseDoubleMatrix2D</tt> the new matrix must also be of type
     * <tt>SparseDoubleMatrix2D</tt>, etc.
     *
     * In general, the new matrix should have internal parametrization as
     * similar as possible.
     *
     * @param rows the number of rows the matrix shall have.
     * @param columns the number of columns the matrix shall have.
     * @return  a new empty matrix of the same dynamic type.
     */
    public DoubleMatrix2D like(int rows, int columns)
    {
        return new SparseDoubleMatrix2D(rows, columns);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, entirelly independent of the receiver.
     * For example, if the receiver is an instance of type
     * <tt>DenseDoubleMatrix2D</tt> the new matrix must be of type
     * <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type
     * <tt>SparseDoubleMatrix2D</tt> the new matrix must be of type
     * <tt>SparseDoubleMatrix1D</tt>, etc.
     *
     * @param  size the number of cells the matrix shall have.
     * @return  a new matrix of the corresponding dynamic type.
     */
    public DoubleMatrix1D like1D(int size)
    {
        return new SparseDoubleMatrix1D(size);
    }

    /**
     * Construct and returns a new 1-d matrix <i>of the corresponding dynamic
     * type</i>, sharing the same cells.
     * For example, if the receiver is an instance of type
     * <tt>DenseDoubleMatrix2D</tt> the new matrix must be of type
     * <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type
     * <tt>SparseDoubleMatrix2D</tt> the new matrix must be of type
     * <tt>SparseDoubleMatrix1D</tt>, etc.
     *
     * @param size the number of cells the matrix shall have.
     * @param zero the index of the first element.
     * @param stride the number of indexes between any two elements, i.e.
     *        <tt>index(i+1)-index(i)</tt>.
     * @return  a new matrix of the corresponding dynamic type.
     */
    protected DoubleMatrix1D like1D(int size, int zero, int stride)
    {
        // FIXME: If needed.
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * Returns the matrix cell value at coordinate <tt>[row,column]</tt>.
     *
     * <p>Provided with invalid parameters this method may return invalid
     * objects without throwing any exception.
     * <b>You should only use this method when you are absolutely sure that
     * the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>0 &lt;= column &lt; columns() &amp;&amp;
     * 0 &lt;= row &lt; rows()</tt>.
     *
     * @param row     the index of the row-coordinate.
     * @param column  the index of the column-coordinate.
     * @return the value at the specified coordinate.
     */
    public double getQuick(int row, int column)
    {
        return viewRow(row).getQuick(column);
    }

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the
     * specified value.
     *
     * <p>Provided with invalid parameters this method may access illegal
     * indexes without throwing any exception.
     * <b>You should only use this method when you are absolutely sure that
     * the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>0 &lt;= column &lt; columns() &amp;&amp;
     *                               0 &lt;= row &lt; rows()</tt>.
     *
     * @param row     the index of the row-coordinate.
     * @param column  the index of the column-coordinate.
     * @param value   the value to be filled into the specified cell.
     */
    public void setQuick(int row, int column, double value)
    {
        SparseDoubleMatrix1D rowView = (SparseDoubleMatrix1D)viewRow(row);
        rowView.setQuick(column, value);
        _rows[row] = rowView._nodes;
    }

    /**
       Constructs and returns a new <i>slice view</i> representing the
       columns of the given row.
       The returned view is backed by this matrix, so changes in the
       returned view are reflected in this matrix, and vice-versa.
       To obtain a slice view on subranges, construct a sub-ranging
       view (<tt>viewPart( ...)</tt>), then apply this method to the
       sub-range view.

       <p> <b>Example:</b> <table border="0"> <tr nowrap>
       <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6</td>
       <td>viewRow(0) ==&gt;</td>
       <td valign="top">Matrix1D of size 3:<br> 1, 2, 3</td> </tr> </table>

       @param row the row to fix.
       @return a new slice view.
       @throws IndexOutOfBoundsException if <tt>row &lt; 0 ||
               row &gt;= rows()</tt>.
       @see #viewColumn(int)
    */
    public DoubleMatrix1D viewRow(int row)
    {
        checkRow(row);
        if (_rowViews[row] == null) {
            _rowViews[row] = new SparseDoubleMatrix1D(columns, _rows[row]);
        }
        // FIXME?! Make sure this row is what's actually in the matrix.
        //         This will actually be one step behind when copying
        //         rows between matrices since this view is updated after
        //         it is fetched.
        _rows[row] = _rowViews[row]._nodes;
        return _rowViews[row];
    }

    /**
     * Replaces one row of the matrix.
     *
     * @param row      the row to replace.
     * @param indices  the indices to be filled in the new row.
     * @param values   the values to be filled into the new row.
     */
    public void setRow(int      row,
                       int[]    indices,
                       double[] values)
    {
        checkRow(row);
        SparseDoubleMatrix1D newRow =
            new SparseDoubleMatrix1D(columns, indices, values);
        _rows[row] = newRow._nodes;
        if (_rowViews[row] != null) {
            _rowViews[row]._nodes = newRow._nodes;
        }
    }

    /**
     * Construct and returns a new selection view.
     *
     * @param rowOffsets the offsets of the visible elements.
     * @param columnOffsets the offsets of the visible elements.
     * @return a new view.
     */
    protected DoubleMatrix2D viewSelectionLike(int[] rowOffsets,
                                               int[] columnOffsets)
    {
        // FIXME: If needed.
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * Sets up a matrix with a given number of rows and columns.
     * @param rows the number of rows the matrix shall have.
     * @param columns the number of columns the matrix shall have.
     * @throws IllegalArgumentException if <tt>rows&lt;0 || columns&lt;0 ||
     *         (double)columns*rows &gt; Integer.MAX_VALUE</tt>.
     */
    protected void setUp(int rows, int columns)
    {
        super.setUp(rows, columns);
        _rows = new Feature[rows][];
        _rowViews = new SparseDoubleMatrix1D[rows];
    }
}
