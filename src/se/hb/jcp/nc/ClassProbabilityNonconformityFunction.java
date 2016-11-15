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
 * A nonconformity function based on the predicted class probabilities given
 * by a classifier.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class ClassProbabilityNonconformityFunction
    extends ClassifierNonconformityFunctionBase
    implements java.io.Serializable
{

    public ClassProbabilityNonconformityFunction(double[] classes)
    {
        this(classes, new se.hb.jcp.bindings.libsvm.SVMClassifier());
    }

    public ClassProbabilityNonconformityFunction
               (double[] classes,
                IClassProbabilityClassifier classifier)
    {
        super(classes, classifier);
    }

    @Override
    public IClassificationNonconformityFunction fitNew(DoubleMatrix2D x,
                                                       double[] y)
    {
        ClassProbabilityNonconformityFunction ncf =
            new ClassProbabilityNonconformityFunction
                    (_classes,
                     (IClassProbabilityClassifier)_model.fitNew(x, y));
        return ncf;
    }

    @Override
    public double calculateNonConformityScore(DoubleMatrix1D x, double y)
    {
        double[] probability = new double[_n_classes];
        ((IClassProbabilityClassifier)_model).predict(x, probability);

        double nc = 1.0 - probability[_class_index.get(y)];
        if (DEBUG) {
            System.err.println("  instance (" + x + ") target " + y +
                               ": " + nc);
        }
        double label = _model.predict(x);
        if (probability[_class_index.get(label)] <
            probability[_classes.length - 1 - _class_index.get(label)]) {
            System.err.println("Warning! Poor model prediction (" +
                               label + ") - model label probability (" +
                               probability[_class_index.get(label)] +
                               ") match!");
        }

        return nc;
    }

    @Override
    CalcNCActionBase createNewCalcNCAction(DoubleMatrix2D x,
                                           double[] y,
                                           double[] nc,
                                           int first, int last)
    {
        return new CalcNCAction(x, y, nc, first, last);
    }

    class CalcNCAction extends CalcNCActionBase
    {
        double[] _probability;

        public CalcNCAction(DoubleMatrix2D x,
                            double[] y,
                            double[] nc,
                            int first, int last)
        {
            super(x, y, nc, first, last);
        }

        @Override
        protected void initialize(int first, int last)
        {
            _probability = new double[_n_classes];
        }

        @Override
        protected void finalize(int first, int last)
        {
            _probability = null;
        }

        @Override
        protected void compute(int i)
        {
            // Overriden with inlined computation to avoid reallocating the
            // _probability array for each instance.
            DoubleMatrix1D instance = _x.viewRow(i);
            ((IClassProbabilityClassifier)_model).predict(instance,
                                                          _probability);
            _nc[i] = 1.0 - _probability[_class_index.get(_y[i])];
        }
    }
}
