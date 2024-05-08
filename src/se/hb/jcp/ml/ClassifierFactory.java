// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2015  Anders Gidenstam
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
package se.hb.jcp.ml;

import org.json.JSONObject;

/**
 * Singleton factory for JCP classifiers.
 *
 * @author anders.gidenstam(at)hb.se
 */

public final class ClassifierFactory
{
    private static final ClassifierFactory _theInstance =
        new ClassifierFactory();
    private static final String[] _classifierNames =
        {
            "se.hb.jcp.bindings.libsvm.SVMClassifier",
            "se.hb.jcp.bindings.jlibsvm.SVMClassifier",
            "se.hb.jcp.bindings.jliblinear.LinearClassifier",
            "se.hb.jcp.bindings.deeplearning4j.NNClassifier"
        };

    private ClassifierFactory()
    {
    }

    public String[] getClassifierTypes()
    {
        return _classifierNames;
    }

    public IClassifier createClassifier(int type)
    {
        JSONObject config = new JSONObject();
        return createClassifier(type, config);
    }

    public IClassifier createClassifier(int type, JSONObject config)
    {
        switch (type) {
        case 0:
            return new se.hb.jcp.bindings.libsvm.SVMClassifier(config);
        case 1:
            return new se.hb.jcp.bindings.jlibsvm.SVMClassifier(config);
        case 2:
            return new se.hb.jcp.bindings.jliblinear.LinearClassifier(config);
        case 3:
            return new se.hb.jcp.bindings.deeplearning4j.NNClassifier(config);
        default:
            throw new UnsupportedOperationException("Unknown classifier type.");
        }
    }

    public static ClassifierFactory getInstance()
    {
        return _theInstance;
    }
}
