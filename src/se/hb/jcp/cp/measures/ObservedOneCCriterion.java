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
 * The Observed OneC criterion is the fraction of predictions that only have
 * the true label in their label set.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class ObservedOneCCriterion
    extends AbstractSignificanceBasedMeasure
    implements IObservedMeasure
{
    /**
     * Creates an Observed OneC measure.
     * @param significanceLevel the significance level used for the label sets.
     */
    public ObservedOneCCriterion(double significanceLevel)
    {
        super("Observed OneC accuracy", significanceLevel);
    }

    /**
     * Computes the Observed OneC criterion measure for this prediction and
     * true label.
     * @param prediction   the prediction.
     * @param trueLabel    the true label of the instance.
     * @return the Observed OneC measure for this prediction. Large values are preferable.
     */
    @Override
    public double compute(ConformalClassification prediction,
                          double trueLabel)
    {
        return
            (prediction.getLabelSet(_significanceLevel).size() == 1 &&
             prediction.getLabelSet(_significanceLevel).contains(trueLabel))
            ? 1.0 : 0.0;
    }
}
