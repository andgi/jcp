// JCP - Java Conformal Prediction framework
// Copyright (C) 2018  Anders Gidenstam
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
package se.hb.jcp.cp;

import cern.colt.matrix.DoubleMatrix1D;

/**
 * Represents a multi-probabilistic prediction made by a conformal classifier
 * with bivariate isotonic regression.
 * See [C. Zhou, "Conformal and Venn Predictors for Multi-probabilistic
 * Predictions and Their Applications", Ph.D. Thesis, Department of Computer
 * Science, Royal Holloway, University of London, 2015] for the definitions
 * used here.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class ConformalMultiProbabilisticClassification
    extends ConformalClassification
{
    private final double lower;
    private final double upper;

    public ConformalMultiProbabilisticClassification(IConformalClassifier source,
                                                     DoubleMatrix1D pValues,
                                                     double lower,
                                                     double upper)
    {
        super(source, pValues);
        this.lower = lower;
        this.upper = upper;
    }

    /**
     * Returns the lower end of the probabilistic interval for the class/label
     * point prediction.
     *
     * @return the lower end of the probabilistic interval for the class/label point prediction.
     */
    public double getPointPredictionLowerBoundProbability()
    {
        return lower;
    }

    /**
     * Returns the upper end of the probabilistic interval for the class/label
     * point prediction.
     *
     * @return the upper end of the probabilistic interval for the class/label point prediction.
     */
    public double getPointPredictionUpperBoundProbability()
    {
        return upper;
    }
}
