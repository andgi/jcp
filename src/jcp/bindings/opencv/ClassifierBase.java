// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.opencv;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.opencv.ml.CvStatModel;

import jcp.ml.IClassifier;

abstract class ClassifierBase
    implements IClassifier,
               java.io.Serializable
{
    private static final boolean DEBUG = true;
    private static final DenseDoubleMatrix1D _storageTemplate =
        new DenseDoubleMatrix1D(0);

    protected CvStatModel _model;
    protected int _attributeCount = -1;

    protected DenseDoubleMatrix2D asDDM2D(DoubleMatrix2D x)
    {
        DenseDoubleMatrix2D tmp_x;
        if ((_attributeCount < 0 || x.columns() == _attributeCount) &&
            x instanceof jcp.bindings.opencv.DenseDoubleMatrix2D) {
            tmp_x = (DenseDoubleMatrix2D)x;
        } else {
            if (DEBUG) {
                System.out.println
                    ("jcp.bindings.opencv.ClassifierBase.asDDM2D(): " +
                     "slow path.");
            }
            if (0 <= _attributeCount && x.columns() != _attributeCount) {
                // Truncate/extend as needed. The mismatch isn't necessarily
                // an error when sparse data is used.
                if (DEBUG) {
                    System.out.println
                        ("jcp.bindings.opencv.ClassifierBase.asDDM2D(): " +
                         "The number of attributes does not match: " +
                         "model " + _attributeCount +
                         "; data " + x.columns() + ".");
                }
                tmp_x = new DenseDoubleMatrix2D(x.rows(), _attributeCount);
                for (int r = 0; r < x.rows(); r++) {
                    for (int c = 0;
                         c < Math.min(_attributeCount, x.columns());
                         c++) {
                        tmp_x.setQuick(r, c, x.getQuick(r, c));
                    }
                }
            } else {
                tmp_x = new DenseDoubleMatrix2D(x.rows(), x.columns());
                tmp_x.assign(x);
            }
        }
        return tmp_x;
    }

    protected DenseDoubleMatrix1D asDDM1D(DoubleMatrix1D x)
    {
        DenseDoubleMatrix1D tmp_x;
        if ((_attributeCount < 0 || x.size() == _attributeCount) &&
            x instanceof jcp.bindings.opencv.DenseDoubleMatrix1D) {
            tmp_x = (DenseDoubleMatrix1D)x;
        } else {
            if (DEBUG) {
                System.out.println
                    ("jcp.bindings.opencv.ClassifierBase.asDDM1D(): " +
                     "slow path.");
            }
            if (0 <= _attributeCount && x.size() != _attributeCount) {
                // Truncate/extend as needed. The mismatch isn't necessarily
                // an error when sparse data is used.
                if (DEBUG) {
                    System.out.println
                        ("jcp.bindings.opencv.ClassifierBase.asDDM1D(): " +
                         "The number of attributes does not match: " +
                         "model " + _attributeCount +
                         "; data " + x.size() + ".");
                }
                tmp_x = new DenseDoubleMatrix1D(_attributeCount);
                for (int c = 0;
                     c < Math.min(_attributeCount, x.size());
                     c++) {
                    tmp_x.setQuick(c, x.getQuick(c));
                }
            } else {
                tmp_x = new DenseDoubleMatrix1D(x.size());
                tmp_x.assign(x);
            }
        }
        return tmp_x;
    }

    protected DenseDoubleMatrix1D asDDM1D(double[] y)
    {
        DenseDoubleMatrix1D tmp_y = new DenseDoubleMatrix1D(y.length);
        tmp_y.assign(y);
        return tmp_y;
    }

    public int getAttributeCount()
    {
        return _attributeCount;
    }

    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }

    protected abstract CvStatModel getNewInstance();

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        // Create a (likely) unique file name for the OpenCV model.
        String fileName =
            Long.toHexString(Double.doubleToLongBits(Math.random())) +
            ".opencv";

        // Write the OpenCV model to a separate file.
        _model.save(fileName);

        // Write the attribute count and OpenCV model file name to the
        // Java output stream.
        oos.writeObject(_attributeCount);
        oos.writeObject(fileName);
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        // Load the attribute count.
        _attributeCount = (int)ois.readObject();
        // Load OpenCV model file name from the Java input stream.
        String fileName = (String)ois.readObject();

        // Load the OpenCV model from the designated file.
        _model = getNewInstance();
        _model.load(fileName);
    }
}
