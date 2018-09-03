// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2016, 2018  Anders Gidenstam
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
 * A base class for nonconformity functions based on the predicted class
 * probabilities given by a classifier.
 *
 * @author anders.gidenstam(at)hb.se
 */
public abstract class ClassProbabilityNonconformityFunctionBase
    extends ClassifierNonconformityFunctionBase
    implements java.io.Serializable
{

    public ClassProbabilityNonconformityFunctionBase
               (double[] classes,
                IClassProbabilityClassifier classifier)
    {
        super(classes, classifier);
    }

    @Override
    public final double calculateNonConformityScore(DoubleMatrix1D x, double y)
    {
        double[] probability = new double[_n_classes];
        double label =
            ((IClassProbabilityClassifier)_model).predict(x, probability);

        double nc = computeNCScore(x, y, probability);
        if (DEBUG) {
            System.err.println("  instance (" + x + ") target " + y +
                               ": " + nc);
        }
        // FIXME: This safety check only works for 2 classes.
        if (_classes.length == 2 &&
            probability[_class_index.get(label)] <
            probability[_classes.length - 1 - _class_index.get(label)]) {
            System.err.println("Warning! Poor model prediction (" +
                               label + ") - model label probability (" +
                               probability[_class_index.get(label)] +
                               ") match!");
        }
        return nc;
    }

    /**
     * Step in the calculateNonConformityScore template method for computing
     * the non-conformity score of an instance based on its assumed label and
     * its class probabilities.
     *
     * @param x            the attributes of the instance.
     * @param y            the assumed label of the instance.
     * @param probability  an double[] array with the instance's class probabilities.
     * @return  the non-conformity score of the instance.
     */
    abstract double computeNCScore(DoubleMatrix1D x, double y,
                                   double[] probability);

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
        protected final void initialize(int first, int last)
        {
            _probability = new double[_n_classes];
        }

        @Override
        protected final void finalize(int first, int last)
        {
            _probability = null;
        }

        @Override
        protected final void compute(int i)
        {
            // Overriden with inlined computation to avoid reallocating the
            // _probability array for each instance.
            DoubleMatrix1D instance = _x.viewRow(i);
            ((IClassProbabilityClassifier)_model).predict(instance,
                                                          _probability);
            _nc[i] = computeNCScore(instance, _y[i], _probability);
        }
    }
}
