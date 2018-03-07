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
package se.hb.jcp.util;

import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Real value indexed 2D matrix.
 * @param <V> the element type.
 *
 * @author anders.gidenstam@hb.se
 */
public class RealIndexedMatrix2D<V extends Object>
    implements java.io.Serializable
{
    private final SortedMap<Double, SortedMap<Double, V>> _m
        = new TreeMap<Double, SortedMap<Double, V>>();

    public int size()
    {
        int size = 0;
        for (SortedMap<Double, V> row : _m.values()) {
            size += row.size();
        }
        return size;
    }

    public boolean isEmpty()
    {
        for (SortedMap<Double, V> row : _m.values()) {
            if (!row.isEmpty()) {
                return false;
            }
        }
        return true;
    }

    public boolean contains(double row, double column)
    {
        if (_m.containsKey(row)) {
            SortedMap<Double, V> r = _m.get(row);
            return r.containsKey(column);
        }
        return false;
    }

    public V get(double row, double column)
    {
        if (_m.containsKey(row)) {
            SortedMap<Double, V> r = _m.get(row);
            return r.get(column);
        }
        return null;
    }

    public V getOrDefault(double row, double column,
                          V defaultValue)
    {
        V result = get(row, column);
        return result != null ? result : defaultValue;
    }

    public Set<Double> getRowIndices()
    {
        return _m.keySet();
    }

    public Set<Double> getColumnIndices(double row)
    {
        if (_m.containsKey(row)) {
            return _m.get(row).keySet();
        } else {
            return new TreeSet<Double>();
        }
    }

    public V getLower(double row, double column)
    {
        SortedMap<Double, V> rowLower = null;
        for (Map.Entry<Double, SortedMap<Double, V>> e : _m.entrySet()) {
            if (e.getKey() <= row) {
                rowLower = e.getValue();
            } else {
                break;
            }
        }
        if (rowLower == null) {
            return null;
        } else {
            V value = null;
            for (Map.Entry<Double, V> e : rowLower.entrySet()) {
                if (e.getKey() <= column) {
                    value = e.getValue();
                } else {
                    break;
                }
            }
            return value;
        }
    }

    public V getUpper(double row, double column)
    {
        SortedMap<Double, V> rowUpper = null;
        for (Map.Entry<Double, SortedMap<Double, V>> e : _m.entrySet()) {
            if (e.getKey() <= row) {
                rowUpper = e.getValue();
            } else {
                rowUpper = e.getValue();
                break;
            }
        }
        if (rowUpper == null) {
            return null;
        } else {
            V value = null;
            for (Map.Entry<Double, V> e : rowUpper.entrySet()) {
                if (e.getKey() <= column) {
                    value = e.getValue();
                } else {
                    value = e.getValue();
                    break;
                }
            }
            return value;
        }
    }

    public void put(double row, double column, V value)
    {
        SortedMap<Double, V> r;
        if (_m.containsKey(row)) {
            r = _m.get(row);
        } else {
            r = new TreeMap<Double, V>();
            _m.put(row, r);
        }
        r.put(column, value);
    }

    public void clear()
    {
        for (SortedMap<Double, V> row : _m.values()) {
            row.clear();
        }
        _m.clear();
    }

    @Override
    public String toString()
    {
        StringBuilder result = new StringBuilder();
        SortedMap<Double, V> row = null;
        for (Map.Entry<Double, SortedMap<Double, V>> re : _m.entrySet()) {
            result = result.append(re.getKey()).append(":{");
            for (Map.Entry<Double, V> ce : re.getValue().entrySet()) {
                result = result.append(ce.getKey()).append(":")
                             .append(ce.getValue()).append("  ");
            }
            result = result.append("}\n");
        }
        return result.toString();
    }
}
