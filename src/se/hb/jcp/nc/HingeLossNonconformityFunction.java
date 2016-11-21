// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2016  Anders Gidenstam
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
package se.hb.jcp.nc;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import se.hb.jcp.ml.IClassProbabilityClassifier;

/**
 * A hinge loss nonconformity function based on the predicted class
 * probabilities given by a classifier.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class HingeLossNonconformityFunction
    extends ClassProbabilityNonconformityFunctionBase
    implements java.io.Serializable
{
    public HingeLossNonconformityFunction(double[] classes)
    {
        this(classes, new se.hb.jcp.bindings.libsvm.SVMClassifier());
    }

    public HingeLossNonconformityFunction
               (double[] classes,
                IClassProbabilityClassifier classifier)
    {
        super(classes, classifier);
    }

    @Override
    public IClassificationNonconformityFunction fitNew(DoubleMatrix2D x,
                                                       double[] y)
    {
        return new HingeLossNonconformityFunction
                       (_classes,
                        (IClassProbabilityClassifier)_model.fitNew(x, y));
    }

    @Override
    double computeNCScore(DoubleMatrix1D x, double y,
                          double[] probability)
    {
        return 1.0 - probability[_class_index.get(y)];
    }
}
