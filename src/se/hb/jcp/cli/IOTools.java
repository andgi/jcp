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
package se.hb.jcp.cli;

import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix1D;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.json.JSONWriter;

import se.hb.jcp.cp.ConformalClassification;

/**
 * Utility functions for reading/writing data to/from JSON.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class IOTools
{

    /**
     * Write an instance to a JSON writer.
     *
     * The sparse format is:
     * { "&lt;attribute index&gt;":&lt;attribute value&gt;, ... }
     * and indices are 0-based.
     *
     * @param instance      the instance to write.
     * @param jsonWriter    the JSON writer.
     */
    public static void writeAsJSON(DoubleMatrix1D instance,
                                   JSONWriter     jsonWriter)
    {
        writeAsJSON(instance, false, 0.0, jsonWriter);
    }

    /**
     * Write an instance to a JSON writer.
     *
     * The sparse format is:
     * { "&lt;attribute index&gt;":&lt;attribute value&gt;, ... }
     * and indices are 0-based.
     *
     * @param instance        the instance to write.
     * @param target          the instance target/label.
     * @param jsonWriter      the JSON writer.
     */
    public static void writeAsJSON(DoubleMatrix1D instance,
                                   double         target,
                                   JSONWriter     jsonWriter)
    {
        writeAsJSON(instance, true, target, jsonWriter);
    }

    /**
     * Read an instance from a JSON stream.
     *
     * The sparse format is:
     * { "&lt;attribute index&gt;":&lt;attribute value&gt;, ... }
     * and indices are 0-based.
     *
     * @param instanceReader    the JSON reader.
     * @param instance          the instance to read into.
     * @return true if an instance was successfully read; false otherwise.
     */
    public static boolean readInstanceFromJSON(JSONTokener instanceReader,
                                               DoubleMatrix1D instance)
    {
        try {
            JSONObject jsonInstance = new JSONObject(instanceReader);

            instance.assign(0);
            for (Object o : jsonInstance.keySet()) {
                String key = (String)o;
                int index  = Integer.parseInt(key);
                instance.set(index, jsonInstance.getDouble(key));
            }
            return true;
        } catch (JSONException e) {
            return false;
        }
    }

    /**
     * Write a <tt>ConformalClassification</tt> as JSON to a JSON writer.
     *
     * The format is:
     * {
     *     "p-values":{[&lt;label&gt;:&lt;p-value&gt;]*},
     *     "point-prediction":{"label":&lt;label&gt;,
     *                         "confidence":&lt;confidence&gt;,
     *                         "credibility":&lt;credibility&gt;}
     * }.
     *
     * @param prediction    the prediction to write.
     * @param resultWriter  the JSON writer.
     */
    public static void writeAsJSON(ConformalClassification prediction,
                                   JSONWriter              resultWriter)
    {
        resultWriter.object();
        // Write the p-values hash.
        resultWriter.key("p-values");
        resultWriter.object();
        for (int i = 0; i < prediction.getPValues().size(); i++) {
            resultWriter.key("" + prediction.getSource().getLabels()[i]);
            resultWriter.value(prediction.getPValues().get(i));
        }
        resultWriter.endObject();
        // Write the point-prediction hash.
        resultWriter.key("point-prediction");
        resultWriter.object();
        resultWriter.key("label");
        double label = prediction.getLabelPointPrediction();
        if (label != Double.NaN) {
            // Only show the label if it is unique.
            resultWriter.value("" + label);
        }
        resultWriter.key("confidence");
        resultWriter.value(prediction.getPointPredictionConfidence());
        resultWriter.key("credibility");
        resultWriter.value(prediction.getPointPredictionCredibility());
        resultWriter.endObject();

        resultWriter.endObject();
    }

    private static void writeAsJSON(DoubleMatrix1D instance,
                                    boolean        includeTarget,
                                    double         target,
                                    JSONWriter     jsonWriter)
    {
        IntArrayList indices   = new IntArrayList();
        DoubleArrayList values = new DoubleArrayList();
        jsonWriter.object();
        instance.getNonZeros(indices, values);
        for (int i = 0; i < indices.size(); i++) {
            jsonWriter.key("" + indices.get(i));
            jsonWriter.value(values.get(i));
        }
        if (includeTarget) {
            jsonWriter.key("" + instance.size());
            jsonWriter.value(target);
        }
        jsonWriter.endObject();
    }

}
