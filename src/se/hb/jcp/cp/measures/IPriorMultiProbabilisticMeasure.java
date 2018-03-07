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
package se.hb.jcp.cp.measures;

import se.hb.jcp.cp.ConformalMultiProbabilisticClassification;

/**
 * A prior measure depends only on the prediction made.
 * See [C. Zhou, "Conformal and Venn Predictors for Multi-probabilistic
 * Predictions and Their Applications", Ph.D. Thesis, Department of Computer
 * Science, Royal Holloway, University of London, 2015] for the definitions
 * used here.
 *
 * @author anders.gidenstam(at)hb.se
 */
public interface IPriorMultiProbabilisticMeasure extends IPriorMeasure
{
    /**
     * Compute the measure for the supplied conformal prediction.
     * @param prediction   a <tt>ConformalMultiProbabilisticClassification</tt>.
     * @return             the measure for the supplied prediction.
     */
    double compute(ConformalMultiProbabilisticClassification prediction);
}
