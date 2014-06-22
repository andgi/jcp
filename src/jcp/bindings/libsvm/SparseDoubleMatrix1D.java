// Copyright (C) 2014  Anders Gidenstam
// The public interface is based on cern.colt.matrix.DoubleMatrix1D.
// License: to be defined.
package jcp.bindings.libsvm;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;

/**
 * Class for sparse 1-d matrices (aka <i>vectors</i>) holding
 * <tt>double</tt> elements in the sparse format expected by the C
 * library libsvm. See the documentation for libsvm for more details.
 *
 * @author anders.gidenstam(at)hb.se
*/
// TODO: Make sure to adhere to colt's conventions.

public class SparseDoubleMatrix1D extends cern.colt.matrix.DoubleMatrix1D
{
    /**
     * C-side pointer to a pointer to an array of svm_nodes storing the matrix
     * contents.
     */
    long Cptr;

    /**
     * Constructs a matrix with a copy of the given values.
     * The values are copied. So subsequent changes in <tt>values</tt>
     * are not reflected in the matrix, and vice-versa.
     *
     * @param values the values to be filled into the new matrix.
     */
    public SparseDoubleMatrix1D(double[] values) {
        int[] indices = new int[values.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        setUp(values.length);
        Cptr = native_vector_create_from(indices, values);
    }

    /**
     * Constructs a matrix with a copy of the given values.
     * The values are copied. So subsequent changes in <tt>values</tt>
     * are not reflected in the matrix, and vice-versa.
     * All other entries are initially <tt>0</tt>.
     *
     * @param columns  the number of columns the matrix shall have.
     * @param indices  the indices to be filled in the new matrix.
     * @param values   the values to be filled into the new matrix.
     */
    public SparseDoubleMatrix1D(int      columns,
                                int[]    indices,
                                double[] values) {
        setUp(columns);
        Cptr = native_vector_create_from(indices, values);
    }

    /**
     * Constructs a matrix with a given number of columns.
     * All entries are initially <tt>0</tt>.
     * @param columns the number of columns the matrix shall have.
     * @throws IllegalArgumentException if
               <tt>columns<0 || columns > Integer.MAX_VALUE</tt>.
    */
    public SparseDoubleMatrix1D(int columns) {
        setUp(columns);
        Cptr = native_vector_create(columns);
    }

    SparseDoubleMatrix1D(int columns, long Cptr) {
        setUp(columns);
        isNoView = false;
        this.Cptr = Cptr;
    }

    /**
     * Replaces all cell values of the receiver with the values of
     * another matrix.  Both matrices must have the same size.
     * If both matrices share the same cells (as is the case if they
     * are views derived from the same matrix) and intersect in an
     * ambiguous way, then replaces <i>as if</i> using an intermediate
     * auxiliary deep copy of <tt>other</tt>.
     *
     * @param     other   the source matrix to copy from (may be identical to the receiver).
     * @return <tt>this</tt> (for convenience only).
     * @throws      IllegalArgumentException if <tt>size() != other.size()</tt>.
     */
    public DoubleMatrix1D assign(DoubleMatrix1D other) {
        if (other==this) {
            return this;
        }
        checkSize(other);
        if (other instanceof jcp.bindings.libsvm.SparseDoubleMatrix1D) {
            native_vector_assign(this.Cptr, ((SparseDoubleMatrix1D)other).Cptr);
            return this;
        } else {
            IntArrayList indexList = new IntArrayList();
            DoubleArrayList valueList = new DoubleArrayList();
            other.getNonZeros(indexList, valueList);
            // FIXME: This will leak the base pointer storage for the temporary
            //        matrix.
            native_vector_assign
                (this.Cptr, native_vector_create_from(indexList.elements(),
                                                      valueList.elements()));
            return this;
        }
}

    /**
     * Construct and returns a new empty matrix <i>of the same dynamic
     * type</i> as the receiver, having the specified size.
     * For example, if the receiver is an instance of type
     * <tt>DenseDoubleMatrix1D</tt> the new matrix must also be of
     * type <tt>DenseDoubleMatrix1D</tt>, if the receiver is an
     * instance of type <tt>SparseDoubleMatrix1D</tt> the new matrix
     * must also be of type <tt>SparseDoubleMatrix1D</tt>, etc.  In
     * general, the new matrix should have internal parametrization as
     * similar as possible.
     *
     * @param size  the number of cell the matrix shall have.
     * @return  a new empty matrix of the same dynamic type.
     */
    public DoubleMatrix1D like(int size) {
        return new SparseDoubleMatrix1D(size);
    }

    /**
     * Construct and returns a new 2-d matrix <i>of the corresponding
     * dynamic type</i>, entirelly independent of the receiver.
     * For example, if the receiver is an instance of type
     * <tt>DenseDoubleMatrix1D</tt> the new matrix must be of type
     * <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of
     * type <tt>SparseDoubleMatrix1D</tt> the new matrix must be of
     * type <tt>SparseDoubleMatrix2D</tt>, etc.
     *
     * @param rows the number of rows the matrix shall have.
     * @param columns the number of columns the matrix shall have.
     * @return  a new matrix of the corresponding dynamic type.
     */
    public DoubleMatrix2D like2D(int rows, int columns) {
        return new SparseDoubleMatrix2D(rows, columns);
    }

    /**
     * Returns the matrix cell value at coordinate <tt>column</tt>.
     *
     * <p>Provided with invalid parameters this method may return invalid
     * objects without throwing any exception.
     * <b>You should only use this method when you are absolutely sure that
     * the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>0 &lt;= column &lt; size()</tt>.
     *
     * @param column  the index of the column-coordinate.
     * @return the value at the specified coordinate.
     */
    public double getQuick(int column) {
        return native_vector_get(Cptr, column);
    }

    /**
     * Sets the matrix cell at coordinate <tt>index</tt> to the
     * specified value.
     *
     * <p>Provided with invalid parameters this method may access
     * illegal indexes wi thout throwing any exception.
     * <b>You should only use this method when you are absolutely sure that
     * the coordinate is within bounds.</b>
     * Precondition (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
     *
     * @param index  the index of the cell.
     * @param value  the value to be filled into the specified cell.
     */
    public void setQuick(int index, double value) {
        native_vector_set(Cptr, index, value);
    }

    /**
     * Construct and returns a new selection view.
     *
     * @param offsets  the offsets of the visible elements.
     * @return a new view.
     */
    protected DoubleMatrix1D viewSelectionLike(int[] offsets) {
        // FIXME: If needed.
        throw new UnsupportedOperationException("Not implemented");
    }

    protected void finalize() throws Throwable {
        if (Cptr != 0 && isNoView) {
            native_vector_free(Cptr);
            Cptr = 0;
        }
    }

    // Internal native functions.
    private static native long native_vector_create(int size);
    private static native void native_vector_free(long ptr);
    private static native long native_vector_create_from(int[]    columns,
                                                         double[] values);
    private static native void native_vector_assign(long this_ptr,
                                                    long source_ptr);
    private static native double native_vector_get(long ptr,
                                                   int column);
    private static native void native_vector_set(long ptr,
                                                 int column,
                                                 double value);

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
