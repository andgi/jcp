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
import java.util.Date;
import java.util.Random;

public class Util
{
    private static Random _random = new Random(new Date().getTime());

    @Deprecated
    public static double[] calc_p_values(double[] nc_pred, double[] nc_cal)
    {
        double[] p_values = new double[nc_pred.length];
        for (int i = 0; i < p_values.length; i++) {
            int idx = Arrays.binarySearch(nc_cal, nc_pred[i]);           
            if (idx < 0) {
                p_values[i] = (nc_cal.length + (idx + 1) + 1) / (nc_cal.length + 1.0);
            } else {
                int count = 0;
                for (int j = idx + 1; j < nc_cal.length && nc_pred[i] == nc_cal[j]; j++) { count++; }
                for (int j = idx; j >= 0 && nc_pred[i] == nc_cal[j]; j--) { count++;  idx--;}
                double p = (nc_cal.length - (idx + 1) - count) / (nc_cal.length + 1.0);
                p_values[i] = p + ((count + 1) * (_random.nextDouble())) / (nc_cal.length + 1.0);
            }
        }
        return p_values;
    }
    
    @Deprecated
    public static boolean[] calc_inclusion(double[] nc_pred, double[] nc_cal, double significance)
    {
        boolean[] include = new boolean[nc_pred.length];
        double[] p_values = calc_p_values(nc_pred, nc_cal);
        for (int i = 0; i < p_values.length; i++) {
            include[i] = (p_values[i] >= significance);
        }
        return include;
    }

    public static double calculatePValue(double nc_pred, double[] nc_cal)
    {
        double p_value;
        int idx = Arrays.binarySearch(nc_cal, nc_pred);
        if (idx < 0) {
            p_value = (nc_cal.length + (idx + 1) + 1) / (nc_cal.length + 1.0);
        } else {
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
                idx--;
            }
            double p =
                (nc_cal.length - (idx + 1) - count) / (nc_cal.length + 1.0);
            p_value =
                p + ((count + 1) * (_random.nextDouble())) /
                (nc_cal.length + 1.0);
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
