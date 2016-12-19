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
 * Maintains a running average of an observed measure.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class AggregatedObservedMeasure
{
    IObservedMeasure _measure;
    int    _n;
    double _sum;

    /**
     * Creates an aggregating observed measure from the single prediction one.
     * @param measure the observed measure.
     */
    public AggregatedObservedMeasure(IObservedMeasure measure)
    {
        _measure = measure;
        _n = 0;
        _sum = 0;
    }

    /**
     * Adds the supplied conformal prediction to the aggregated measure.
     * @param prediction   a <tt>ConformalClassification</tt>.
     * @param trueLabel    the true label of the instance.
     */
    public void add(ConformalClassification prediction, double trueLabel)
    {
        _n++;
        _sum += _measure.compute(prediction, trueLabel);
    }

    /**
     * Gets the current number of observations of the measure.
     * @return the current number of observations.
     */
    public int getNumberOfObservations()
    {
        return _n;
    }

    /**
     * Gets the current mean of the aggregated observed measure.
     * @return the current mean of the aggregated observed measure.
     */
    public double getMean()
    {
        return _sum / _n;
    }

    @Override
    public String toString()
    {
        return _measure.getName() + ": " + getMean();
    }
}
