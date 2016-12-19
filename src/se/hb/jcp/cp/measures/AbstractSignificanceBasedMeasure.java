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
 * Base class for measures that depend on the significance level.
 *
 * @author anders.gidenstam(at)hb.se
 */
public class AbstractSignificanceBasedMeasure implements IMeasure
{
    private final String _name;
    final double _significanceLevel;

    /**
     * Creates an abstract significance based measure.
     * @param measureName       the base name of the measure.
     * @param significanceLevel the significance level used for the label sets.
     */
    public AbstractSignificanceBasedMeasure(String measureName,
                                            double significanceLevel)
    {
        _name =
            measureName + "(significanceLevel = " + significanceLevel + ")";
        _significanceLevel = significanceLevel;
    }

    /**
     * Get the name of this measure.
     * @return the name of this measure.
     */
    @Override
    public String getName()
    {
        return _name;
    }
}
