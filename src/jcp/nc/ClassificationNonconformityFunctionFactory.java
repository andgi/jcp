// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.nc;

/**
 * Singleton factory for JCP classification nonconformity functions.
 *
 * @author anders.gidenstam(at)hb.se
 */

public final class ClassificationNonconformityFunctionFactory
{
    private static final
        ClassificationNonconformityFunctionFactory _theInstance =
            new ClassificationNonconformityFunctionFactory();
    private static final String[] _ncfNames =
        {
            "class probability nonconformity function",
            "attribute average nonconformity function"
        };

    private ClassificationNonconformityFunctionFactory()
    {
    }

    public String[] getNonconformityFunctions()
    {
        return _ncfNames;
    }

    public IClassificationNonconformityFunction
        createNonconformityFunction(int type,
                                    double[] classes,
                                    jcp.ml.IClassifier classifier)
    {
        switch (type) {
        case 0:
            return new ClassProbabilityNonconformityFunction(classes,
                                                             classifier);
        case 1:
            return new AverageClassificationNonconformityFunction(classes);
        default:
            throw new UnsupportedOperationException
                ("Unknown nonconformity function type.");
        }
    }

    public static ClassificationNonconformityFunctionFactory getInstance()
    {
        return _theInstance;
    }
}
