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
 * Maintains a running average of a prior measure.
 * See [V. Vovk, V. Fedorova, I. Nouretdinov and A. Gammerman, "Criteria of
 * Efficiency for Conformal Prediction", COPA 2016, LNAI 9653, pp. 23-39, 2016]
 * for the definitions used here.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class AggregatedPriorMeasure
{
    IPriorMeasure _measure;
    int    _n;
    double _sum;

    /**
     * Creates an aggregating prior measure from the single prediction one.
     * @param measure the prior measure.
     */
    public AggregatedPriorMeasure(IPriorMeasure measure)
    {
        _measure = measure;
        _n = 0;
        _sum = 0;
    }

    /**
     * Adds the supplied conformal prediction to the aggregated prior measure.
     * @param prediction   a <tt>ConformalClassification</tt>.
     */
    public void add(ConformalClassification prediction)
    {
        _n++;
        _sum += _measure.compute(prediction);
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
     * Gets the current mean of the aggregated prior measure.
     * @return the current mean of the aggregated prior measure.
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
