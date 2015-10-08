// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2015  Anders Gidenstam
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
package se.hb.jcp.bindings.opencv;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.opencv.core.TermCriteria;
import org.opencv.ml.CvStatModel;

import org.json.JSONObject;

import se.hb.jcp.ml.IClassifier;

abstract class ClassifierBase
    implements IClassifier,
               java.io.Serializable
{
    private static final boolean DEBUG = true;
    private static final DenseDoubleMatrix1D _storageTemplate =
        new DenseDoubleMatrix1D(0);

    protected CvStatModel _model;
    protected JSONObject _jsonParameters;
    protected int _attributeCount = -1;

    protected DenseDoubleMatrix2D asDDM2D(DoubleMatrix2D x)
    {
        DenseDoubleMatrix2D tmp_x;
        if ((_attributeCount < 0 || x.columns() == _attributeCount) &&
            x instanceof se.hb.jcp.bindings.opencv.DenseDoubleMatrix2D) {
            tmp_x = (DenseDoubleMatrix2D)x;
        } else {
            if (DEBUG) {
                System.out.println
                    ("se.hb.jcp.bindings.opencv.ClassifierBase.asDDM2D(): " +
                     "slow path.");
            }
            if (0 <= _attributeCount && x.columns() != _attributeCount) {
                // Truncate/extend as needed. The mismatch isn't necessarily
                // an error when sparse data is used.
                if (DEBUG) {
                    System.out.println
                        ("se.hb.jcp.bindings.opencv.ClassifierBase.asDDM2D():" +
                         " The number of attributes does not match: " +
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
            x instanceof se.hb.jcp.bindings.opencv.DenseDoubleMatrix1D) {
            tmp_x = (DenseDoubleMatrix1D)x;
        } else {
            if (DEBUG) {
                System.out.println
                    ("se.hb.jcp.bindings.opencv.ClassifierBase.asDDM1D(): " +
                     "slow path.");
            }
            if (0 <= _attributeCount && x.size() != _attributeCount) {
                // Truncate/extend as needed. The mismatch isn't necessarily
                // an error when sparse data is used.
                if (DEBUG) {
                    System.out.println
                        ("se.hb.jcp.bindings.opencv.ClassifierBase.asDDM1D():" +
                         " The number of attributes does not match: " +
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

    protected TermCriteria readTerminationCriteria()
    {
        if (_jsonParameters.has("termination_criteria")) {
            JSONObject termination =
                _jsonParameters.getJSONObject("termination_criteria");
            int criteria = 0;
            int max_iter = 0;
            double epsilon = 0.0;
            if (termination.has("max_count")) {
                criteria += TermCriteria.MAX_ITER;
                max_iter = termination.getInt("max_iter");
            }
            if (termination.has("epsilon")) {
                criteria += TermCriteria.EPS;
                epsilon = termination.getDouble("epsilon");
            }
            return new TermCriteria(criteria,
                                    max_iter,
                                    epsilon);
        } else {
            return null;
        }
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        // Save the model parameters.
        if (_jsonParameters != null) {
            oos.writeObject(_jsonParameters.toString());
        } else {
            oos.writeObject(null);
        }
        // Save the attribute count.
        oos.writeObject(_attributeCount);
        // Save the model if it has been trained.
        if (_attributeCount > -1) {
            // Create a (likely) unique file name for the OpenCV model.
            String fileName =
                Long.toHexString(Double.doubleToLongBits(Math.random())) +
                ".opencv";

            // Save the OpenCV model to a separate file.
            _model.save(fileName);
            // Save the model file name.
            oos.writeObject(fileName);
        } else {
            // The model has not been trained.
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        // Load the model parameters.
        String jsonText = (String)ois.readObject();
        if (jsonText != null) {
            _jsonParameters = new JSONObject(jsonText);
        }
        // Load the attribute count.
        _attributeCount = (int)ois.readObject();

        // Load OpenCV model file name from the Java input stream.
        String fileName = (String)ois.readObject();
        if (fileName != null) {
            // Load the OpenCV model from the designated file.
            _model = getNewInstance();
            _model.load(fileName);
        }
    }
}
