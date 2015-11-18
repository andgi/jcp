// JCP - Java Conformal Prediction framework
// Copyright (C) 2014  Henrik Linusson
// Copyright (C) 2015  Anders Gidenstam
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
package se.hb.jcp.cp;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Util
{
    private static final boolean USE_SMOOTHING = true;

    public static double calculatePValue(double nc_pred, double[] nc_cal)
    {
        double p_value;
        int idx = Arrays.binarySearch(nc_cal, nc_pred);
        if (idx < 0) {
            // The key was not found, idx == -insertion_idx - 1 where
            // insertion_idx is the the index of the first element
            // greater than the search key.
            p_value = (nc_cal.length + (idx + 1) + 1) / (nc_cal.length + 1.0);
        } else {
            // idx == one index containing the search key.
            // Count the number of scores equal to nc_pred in nc_cal for
            // smoothing according to [Vovk, ALRW WP#5, 2012] and place
            // idx on the first score equal to nc_pred.
            int count = 0;
            for (int j = idx + 1;
                 j < nc_cal.length && nc_pred == nc_cal[j];
                 j++) {
                count++;
            }
            for (int j = idx;
                 j >= 0 && nc_pred == nc_cal[j];
                 j--) {
                count++;
                idx = j;
            }
            if (USE_SMOOTHING) {
                // Smoothed p-value according to [Vovk, ALRW WP#5, 2012].
                double theta = ThreadLocalRandom.current().nextDouble(1.0);
                p_value =
                    ((nc_cal.length - idx - count) + theta * (count + 1))
                    / (nc_cal.length + 1.0);
            } else {
                // Unsmoothed p-value.
                p_value = (nc_cal.length - idx + 1) / (nc_cal.length + 1.0);
            }
        }
        return p_value;
    }

    public static boolean calculateInclusion(double   nc_pred,
                                             double[] nc_cal,
                                             double   significance)
    {
        double p_value = calculatePValue(nc_pred, nc_cal);
        return (p_value >= significance);
    }
}
