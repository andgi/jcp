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

    protected DenseDoubleMatrix2D asDDM2D(DoubleMatrix2D x)
    {
        DenseDoubleMatrix2D tmp_x;
        if (x instanceof jcp.bindings.opencv.DenseDoubleMatrix2D) {
            tmp_x = (DenseDoubleMatrix2D)x;
        } else {
            if (DEBUG) {
                System.out.println
                    ("jcp.bindings.opencv.ClassifierBase.asDDM2D(): " +
                     "slow path.");
            }
            tmp_x = new DenseDoubleMatrix2D(x.rows(), x.columns());
            tmp_x.assign(x);
        }
        return tmp_x;
    }

    protected DenseDoubleMatrix1D asDDM1D(DoubleMatrix1D x)
    {
        DenseDoubleMatrix1D tmp_x;
        if (x instanceof jcp.bindings.opencv.DenseDoubleMatrix1D) {
            tmp_x = (DenseDoubleMatrix1D)x;
        } else {
            if (DEBUG) {
                System.out.println
                    ("jcp.bindings.opencv.ClassifierBase.asDDM1D(): " +
                     "slow path.");
            }
            tmp_x = new DenseDoubleMatrix1D(x.size());
            tmp_x.assign(x);
        }
        return tmp_x;
    }

    protected DenseDoubleMatrix1D asDDM1D(double[] y)
    {
        DenseDoubleMatrix1D tmp_y = new DenseDoubleMatrix1D(y.length);
        tmp_y.assign(y);
        return tmp_y;
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

        // Write the OpenCV model file name to the Java output stream.
        oos.writeObject(fileName);
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        // Load OpenCV model file name from the Java input stream.
        String fileName = (String)ois.readObject();

        // Load the OpenCV model from the designated file.
        _model = getNewInstance();
        _model.load(fileName);
    }
}
