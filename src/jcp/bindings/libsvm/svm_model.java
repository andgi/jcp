// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.libsvm;

public class svm_model //implements java.io.Serializable
{
    svm_model(long Cptr)
    {
        this.Cptr = Cptr;
    }

    // C-side pointer to a svm_model.
    long Cptr;
}
