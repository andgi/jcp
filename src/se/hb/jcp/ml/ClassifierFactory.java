// Copyright (C) 2014 - 2015  Anders Gidenstam
// License: to be defined.
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
            "se.hb.jcp.bindings.opencv.SVMClassifier",
            "se.hb.jcp.bindings.opencv.RandomForestClassifier"
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
            return new se.hb.jcp.bindings.opencv.SVMClassifier(config);
        case 4:
            return new se.hb.jcp.bindings.opencv.RandomForestClassifier(config);
        default:
            throw new UnsupportedOperationException("Unknown classifier type.");
        }
    }

    public static ClassifierFactory getInstance()
    {
        return _theInstance;
    }
}
