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
 * The E/Excess criterion is a prior efficiency measure based on how much the
 * size of the label set exceeds 1.
 * See [V. Vovk, V. Fedorova, I. Nouretdinov and A. Gammerman, "Criteria of
 * Efficiency for Conformal Prediction", COPA 2016, LNAI 9653, pp. 23-39, 2016]
 * for the definitions used here.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class ExcessCriterion
    extends AbstractSignificanceBasedMeasure
    implements IPriorMeasure
{
    /**
     * Creates an E/Excess criterion measure.
     * @param significanceLevel the significance level used for the label sets.
     */
    public ExcessCriterion(double significanceLevel)
    {
        super("Excess criterion", significanceLevel);
    }

    /**
     * Computes the E/Excess criterion measure for this prediction.
     * @param prediction the prediction.
     * @return the Excess criterion measure for this prediction. Small values are preferable.
     */
    @Override
    public double compute(ConformalClassification prediction)
    {
        return
            Math.max(0.0,
                     prediction.getClassSet(_significanceLevel).size() - 1.0);
    }
}
