// Copyright (C) 2014  Anders Gidenstam
// The public interface is based on cern.colt.matrix.DoubleMatrix2D.
// License: to be defined.
package jcp.bindings.libsvm;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

/**
 * Class for sparse 2-d matrices holding <tt>double</tt> elements in
 * the sparse format expected by the C library libsvm. See the
 * documentation for libsvm for more details.
 *
 * @author anders.gidenstam(at)hb.se
*/

public class SparseDoubleMatrix2D extends cern.colt.matrix.DoubleMatrix2D
{
    /**
     * C-side pointer to an array of svm_node arrays storing the matrix
     * contents.
     */
    protected long Cptr;
    /**
     * SparseDoubleMatrix1D views of rows in this matrix.
     * These are created and stored on demand.
     */
    protected SparseDoubleMatrix1D[] rowViews;

    /**
     * Constructs a matrix with a given number of rows and columns.
     * All entries are initially <tt>0</tt>.
     * @param rows the number of rows the matrix shall have.
     * @param columns the number of columns the matrix shall have.
     * @throws IllegalArgumentException if
               <tt>rows<0 || columns<0 || (double)columns*rows >
               Integer.MAX_VALUE</tt>.
    */
    public SparseDoubleMatrix2D(int rows, int columns)
    {
        setUp(rows, columns);
        Cptr = native_matrix_create(rows, columns);
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
    public DoubleMatrix2D like(int rows, int columns) {
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
    public DoubleMatrix1D like1D(int size) {
         // FIXME: If needed.
        throw new UnsupportedOperationException("Not implemented");
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
    protected DoubleMatrix1D like1D(int size, int zero, int stride) {
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
     * Precondition (unchecked): <tt>0 &lt;= column &lt; columns() &&
     * 0 &lt;= row &lt; rows()</tt>.
     *
     * @param row     the index of the row-coordinate.
     * @param column  the index of the column-coordinate.
     * @return the value at the specified coordinate.
     */
    public double getQuick(int row, int column) {
        return native_matrix_get(Cptr, row, column);
    }

    /**
     * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the
     * specified value.
     *
     * <p>Provided with invalid parameters this method may access illegal
     * indexes without throwing any exception.
     * <b>You should only use this method when you are absolutely sure that
     * the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>0 &lt;= column &lt; columns() &&
     *                               0 &lt;= row &lt; rows()</tt>.
     *
     * @param row     the index of the row-coordinate.
     * @param column  the index of the column-coordinate.
     * @param value   the value to be filled into the specified cell.
     */
    public void setQuick(int row, int column, double value) {
        if (getQuick(row, column) == 0.0) {
            // FIXME: If needed.
            throw new UnsupportedOperationException("Not implemented");
        } else {
            // The entry exists already.
            native_matrix_set(Cptr, row, column, value);
        }
    }

    /**
     * Returns one row of the matrix as a SparseDoubleMatrix1D.
     *
     * FIXME: The resulting matrix will share storage with this instance.
     *        Must handle freeing eventually.
     *
     * @param row     the row to return.
     * @returns a <tt>SparseDoubleMatrix1D</tt> representing a view of the row.
     */
    public SparseDoubleMatrix1D getRow(int row) {
        if (rowViews[row] == null) {
            rowViews[row] =
                new SparseDoubleMatrix1D(columns,
                                         native_matrix_get_row(Cptr, row));
        }
        return rowViews[row];
    }

    /**
     * Replaces one row of the matrix.
     *
     * FIXME: The old row will be freed. If there are other views of it
     *        they will now reference freed memory. Beware!
     *
     * @param row      the row to replace.
     * @param indices  the indices to be filled in the new row.
     * @param values   the values to be filled into the new row.
     */
    public void setRow(int      row,
                       int[]    indices,
                       double[] values) {
        // FIXME: Verify row index and indices.
        long newRow = native_matrix_set_row(Cptr, row, indices, values);
        if (rowViews[row] != null) {
            rowViews[row].Cptr = newRow;
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
                                               int[] columnOffsets) {
        // FIXME: If needed.
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * Sets up a matrix with a given number of rows and columns.
     * @param rows the number of rows the matrix shall have.
     * @param columns the number of columns the matrix shall have.
     * @throws IllegalArgumentException if <tt>rows<0 || columns<0 ||
     *         (double)columns*rows > Integer.MAX_VALUE</tt>.
     */
    protected void setUp(int rows, int columns) {
        super.setUp(rows,columns);
        rowViews = new SparseDoubleMatrix1D[rows];
    }

    protected void finalize() throws Throwable {
        if (Cptr != 0 && isNoView) {
            native_matrix_free(Cptr, rows);
            Cptr = 0;
        }
    }

    // Internal native functions.
    private static native long native_matrix_create(int rows, int columns);
    private static native void native_matrix_free(long ptr, int rows);
    private static native double native_matrix_get(long ptr,
                                                   int row, int column);
    private static native void native_matrix_set(long ptr,
                                                 int row, int column,
                                                 double value);
    private static native long native_matrix_get_row(long ptr,
                                                     int row);
    private static native long native_matrix_set_row(long ptr,
                                                     int row,
                                                     int[] columns,
                                                     double[] values);

    static {
        // FIXME: It would have been better not to repeat this here and
        //        keep it only in the svm class.
        try {
            System.loadLibrary("svm");
        } catch (UnsatisfiedLinkError e) {
            System.out.println
                ("Could not load libsvm.");
            System.exit(1);
        }
        try {
            System.loadLibrary("svm-jni");
        } catch (UnsatisfiedLinkError e) {
            System.out.println
                ("Could not load native JNI wrapper code for libsvm.");
            System.exit(1);
        }
    }
}
