// JCP - Java Conformal Prediction framework
// Copyright (C) 2016  Anders Gidenstam
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

import se.hb.jcp.ml.ISVMClassifier;
import se.hb.jcp.nc.ClassifierNonconformityFunctionBase;

/**
 * This class implements a nonconformity function based on the signed
 * distance from the separating hyperplane of a two-class SVM classifier.
 * See e.g. [Toccaceli, Nouretdinov, Gammerman, "Conformal Predictors for
 * Compound Activity Prediction", https://arxiv.org/pdf/1603.04506.pdf, 2016].
 *
 * @author anders.gidenstam(at)hb.se
 */
public class SVMDistanceNonconformityFunction
    extends ClassifierNonconformityFunctionBase
    implements java.io.Serializable
{
    public SVMDistanceNonconformityFunction(double[] classes)
    {
        this(classes, new se.hb.jcp.bindings.jlibsvm.SVMClassifier());
    }

    public SVMDistanceNonconformityFunction
               (double[] classes,
                ISVMClassifier classifier)
    {
        super(classes, classifier);
        if (classes.length > 2) {
            throw new UnsupportedOperationException
                          ("The SVMDistanceNonconformityFunction does not " +
                           "support multiclass problems.");
        }
    }

    @Override
    public IClassificationNonconformityFunction fitNew(DoubleMatrix2D x,
                                                       double[] y)
    {
        SVMDistanceNonconformityFunction ncf =
            new SVMDistanceNonconformityFunction
                    (_classes, (ISVMClassifier)_model.fitNew(x, y));
        return ncf;
    }

    @Override
    public double calculateNonConformityScore(DoubleMatrix1D x, double y)
    {
        return -y * ((ISVMClassifier)_model).distanceFromSeparatingPlane(x);
    }
}
