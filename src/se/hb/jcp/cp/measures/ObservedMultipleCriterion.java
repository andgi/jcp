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
package se.hb.jcp.cp.measures;

import se.hb.jcp.cp.ConformalClassification;

/**
 * The OM/Observed Multiple criterion is an observed efficiency measure based on
 * the number of false labels in the label set.
 * See [V. Vovk, V. Fedorova, I. Nouretdinov and A. Gammerman, "Criteria of
 * Efficiency for Conformal Prediction", COPA 2016, LNAI 9653, pp. 23-39, 2016]
 * for the definitions used here.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class ObservedMultipleCriterion
    extends AbstractSignificanceBasedMeasure
    implements IObservedMeasure
{
    /**
     * Creates an OM/Observed Multiple criterion measure.
     * @param significanceLevel the significance level used for the label sets.
     */
    public ObservedMultipleCriterion(double significanceLevel)
    {
        super("Observed Multiple criterion", significanceLevel);
    }

    /**
     * Computes the OM/Observed Multiple criterion measure for this prediction
     * and true label.
     * @param prediction   the prediction.
     * @param trueLabel    the true label of the instance.
     * @return the Observed Multiple criterion measure for this prediction. Small values are preferable.
     */
    @Override
    public double compute(ConformalClassification prediction,
                          double trueLabel)
    {
        int falseLabels =
                prediction.getLabelSet(_significanceLevel).size() -
                (prediction.getLabelSet(_significanceLevel).contains(trueLabel)
                 ? 1 : 0);
        return (falseLabels > 0) ? 1.0 : 0.0;
    }
}
