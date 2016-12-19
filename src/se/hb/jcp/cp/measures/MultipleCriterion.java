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
 * The M/Multiple criterion is a prior efficiency measure based on the size of
 * the label set.
 * See [V. Vovk, V. Fedorova, I. Nouretdinov and A. Gammerman, "Criteria of
 * Efficiency for Conformal Prediction", COPA 2016, LNAI 9653, pp. 23-39, 2016]
 * for the definitions used here.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class MultipleCriterion
    extends AbstractSignificanceBasedMeasure
    implements IPriorMeasure
{
    /**
     * Creates a M/Multiple criterion measure.
     * @param significanceLevel the significance level used for the label sets.
     */
    public MultipleCriterion(double significanceLevel)
    {
        super("Multiple criterion", significanceLevel);
    }

    /**
     * Computes the M/Multiple criterion measure for this prediction.
     * @param prediction the prediction.
     * @return the Multiple criterion measure for this prediction. Small values are preferable.
     */
    @Override
    public double compute(ConformalClassification prediction)
    {
        return
            (prediction.getClassSet(_significanceLevel).size() > 1) ? 1.0 : 0.0;
    }
}
