// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.ml;

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
            "jcp.bindings.libsvm.SVMClassifier",
            "jcp.bindings.opencv.SVMClassifier",
            "jcp.bindings.opencv.RForestClassifier"
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
        // FIXME: Configuration parameters need to be passed in somehow.
        switch (type) {
        case 0:
            return new jcp.bindings.libsvm.SVMClassifier();
        case 1:
            return new jcp.bindings.opencv.SVMClassifier();
        case 2:
            return new jcp.bindings.opencv.RForestClassifier();
        default:
            throw new UnsupportedOperationException("Unknown classifier type.");
        }
    }

    public static ClassifierFactory getInstance()
    {
        return _theInstance;
    }
}
