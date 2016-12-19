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
 * Maintains running averages for a set of observed measures.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class AggregatedObservedMeasures
{
    AggregatedObservedMeasure[] _measures;

    /**
     * Creates the default set of aggregating observed measures.
     */
    public AggregatedObservedMeasures()
    {
        this(new IObservedMeasure[] {
                new ObservedAccuracy(0.10),
                new ObservedAccuracy(0.05),
                new ObservedAccuracy(0.01),
                new ObservedUnconfidenceCriterion(),
                new ObservedFuzzinessCriterion(),
                new ObservedMultipleCriterion(0.10),
                new ObservedMultipleCriterion(0.05),
                new ObservedMultipleCriterion(0.01),
                new ObservedExcessCriterion(0.10),
                new ObservedExcessCriterion(0.05),
                new ObservedExcessCriterion(0.01)
             });
    }

    /**
     * Creates a set of aggregating observed measures from an array of single
     * prediction ones.
     * @param measures the observed measures.
     */
    public AggregatedObservedMeasures(IObservedMeasure[] measures)
    {
        _measures = new AggregatedObservedMeasure[measures.length];
        for (int i = 0; i < measures.length; i++) {
            _measures[i] = new AggregatedObservedMeasure(measures[i]);
        }
    }

    /**
     * Adds the supplied conformal prediction to the aggregated measures.
     * @param prediction   a <tt>ConformalClassification</tt>.
     * @param trueLabel    the true label of the instance.
     */
    public void add(ConformalClassification prediction, double trueLabel)
    {
        for (AggregatedObservedMeasure m : _measures) {
            m.add(prediction, trueLabel);
        }
    }

    /**
     * Gets one of the aggregated observed measures in this set.
     * @param i the index (0 to size()-1) of the measure.
     * @return a <tt>AggregatedObservedMeasure</tt>.
     */
    public AggregatedObservedMeasure getMeasure(int i)
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
