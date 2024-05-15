package se.hb.jcp.bindings.deeplearning4j;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SparseDoubleMatrix2D extends DoubleMatrix2D {
    INDArray[] _rows;
    protected SparseDoubleMatrix1D[] _rowViews;

    public SparseDoubleMatrix2D(int rows, int columns) {
        setUp(rows, columns);
        _rows = new INDArray[rows];
        for (int r = 0; r < rows; r++) {
            _rows[r] = Nd4j.zeros(columns);
        }
    }

    public DoubleMatrix2D like(int rows, int columns) {
        return new SparseDoubleMatrix2D(rows, columns);
    }

    public DoubleMatrix1D like1D(int size) {
        return new SparseDoubleMatrix1D(size);
    }

    protected DoubleMatrix1D like1D(int size, int zero, int stride) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public double getQuick(int row, int column) {
        return _rows[row].getDouble(column);
    }

    public void setQuick(int row, int column, double value) {
        _rows[row].putScalar(column, value);
    }

    public DoubleMatrix1D viewRow(int row) {
        checkRow(row);
        if (_rowViews[row] == null) {
            _rowViews[row] = new SparseDoubleMatrix1D(columns, _rows[row]);
        }
        _rows[row] = _rowViews[row].getNodes();
        return _rowViews[row];
    }

    public void setRow(int row, int[] indices, double[] values) {
        checkRow(row);
        INDArray newRow = Nd4j.zeros(columns);
        for (int i = 0; i < indices.length; i++) {
            newRow.putScalar(indices[i], values[i]);
        }
        _rows[row] = newRow;
        if (_rowViews[row] != null) {
            _rowViews[row].setNodes(newRow);
        }
    }

    protected DoubleMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        throw new UnsupportedOperationException("Not implemented");
    }

    protected void setUp(int rows, int columns) {
        super.setUp(rows, columns);
        _rows = new INDArray[rows];
        _rowViews = new SparseDoubleMatrix1D[rows];
    }
}
