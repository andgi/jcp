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
 * Maintains running averages for a set of prior measures.
 * See [V. Vovk, V. Fedorova, I. Nouretdinov and A. Gammerman, "Criteria of
 * Efficiency for Conformal Prediction", COPA 2016, LNAI 9653, pp. 23-39, 2016]
 * for the definitions used here.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class AggregatedPriorMeasures
{
    AggregatedPriorMeasure[] _measures;

    /**
     * Creates the default set of aggregating prior measures.
     */
    public AggregatedPriorMeasures()
    {
        this(new IPriorMeasure[] {
                new SumCriterion(),
                new NumberCriterion(0.10),
                new NumberCriterion(0.05),
                new NumberCriterion(0.01),
                new FuzzinessCriterion(),
                new UnconfidenceCriterion(),
                new MultipleCriterion(0.10),
                new MultipleCriterion(0.05),
                new MultipleCriterion(0.01),
                new ExcessCriterion(0.10),
                new ExcessCriterion(0.05),
                new ExcessCriterion(0.01),
                new OneCCriterion(0.10),
                new OneCCriterion(0.05),
                new OneCCriterion(0.01)
        });
    }
    /**
     * Creates a set of aggregating prior measures from an array of single
     * prediction ones.
     * @param measures the prior measures.
     */
    public AggregatedPriorMeasures(IPriorMeasure[] measures)
    {
        _measures = new AggregatedPriorMeasure[measures.length];
        for (int i = 0; i < measures.length; i++) {
            _measures[i] = new AggregatedPriorMeasure(measures[i]);
        }
    }

    /**
     * Adds the supplied conformal prediction to the aggregated prior measures.
     * @param prediction   a <tt>ConformalClassification</tt>.
     */
    public void add(ConformalClassification prediction)
    {
        for (AggregatedPriorMeasure m : _measures) {
            m.add(prediction);
        }
    }

    /**
     * Gets one of the aggregated prior measures in this set.
     * @param i the index (0 to size()-1) of the measure.
     * @return a <tt>AggregatedPriorMeasure</tt>.
     */
    public AggregatedPriorMeasure getMeasure(int i)
    {
        return _measures[i];
    }

    /**
     * Returns the number of measures in this set.
     * @return the number of measures in this set.
     */
    public int size()
    {
        return _measures.length;
    }
}
