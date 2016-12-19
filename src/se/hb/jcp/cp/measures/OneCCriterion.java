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
 * The OneC is the fraction of predictions that only have one label in their
 * label set.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class OneCCriterion
    extends AbstractSignificanceBasedMeasure
    implements IPriorMeasure
{
    /**
     * Creates an OneC measure.
     * @param significanceLevel the significance level used for the label sets.
     */
    public OneCCriterion(double significanceLevel)
    {
        super("OneC criterion", significanceLevel);
    }

    /**
     * Computes the OneC criterion measure for this prediction.
     * @param prediction   the prediction.
     * @return the OneC criterion measure for this prediction. Large values are preferable.
     */
    @Override
    public double compute(ConformalClassification prediction)
    {
        return
            (prediction.getLabelSet(_significanceLevel).size() == 1)
            ? 1.0 : 0.0;
    }
}
