package se.hb.jcp.bindings.deeplearning4j;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SparseDoubleMatrix1D extends DoubleMatrix1D {
    private INDArray _nodes;

    public SparseDoubleMatrix1D(double[] values) {
        setUp(values.length);
        _nodes = Nd4j.zeros(values.length);
        for (int i = 0; i < values.length; i++) {
            _nodes.putScalar(i, values[i]);
        }
    }

    public SparseDoubleMatrix1D(int columns, int[] indices, double[] values) {
        setUp(columns);
        _nodes = Nd4j.zeros(columns);
        for (int i = 0; i < indices.length; i++) {
            _nodes.putScalar(indices[i], values[i]);
        }
    }

    public SparseDoubleMatrix1D(int columns) {
        setUp(columns);
        _nodes = Nd4j.zeros(columns);
    }

    SparseDoubleMatrix1D(int columns, INDArray nodes) {
        setUp(columns);
        _nodes = nodes;
    }

    public DoubleMatrix1D assign(DoubleMatrix1D other) {
        if (other == this) {
            return this;
        }
        checkSize(other);
        if (other instanceof SparseDoubleMatrix1D) {
            _nodes = ((SparseDoubleMatrix1D) other)._nodes.dup();
            return this;
        } else {
            IntArrayList indexList = new IntArrayList();
            DoubleArrayList valueList = new DoubleArrayList();
            other.getNonZeros(indexList, valueList);
            _nodes = Nd4j.zeros(size);
            for (int i = 0; i < indexList.size(); i++) {
                _nodes.putScalar(indexList.get(i), valueList.get(i));
            }
            return this;
        }
    }

    public DoubleMatrix1D like(int size) {
        return new SparseDoubleMatrix1D(size);
    }

    public DoubleMatrix2D like2D(int rows, int columns) {
        return new SparseDoubleMatrix2D(rows, columns);
    }

    public double getQuick(int column) {
        return _nodes.getDouble(column);
    }

    public void setQuick(int index, double value) {
        _nodes.putScalar(index, value);
    }

    public INDArray getNodes() {
        return _nodes;
    }

    public void setNodes(INDArray nodes) {
        _nodes = nodes;
    }

    protected DoubleMatrix1D viewSelectionLike(int[] offsets) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
