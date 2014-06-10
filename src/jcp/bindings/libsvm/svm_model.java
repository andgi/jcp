// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.libsvm;

public class svm_model //implements java.io.Serializable
{
    svm_model(long Cptr)
    {
        this.Cptr = Cptr;
    }

    protected void finalize() throws Throwable {
        if (Cptr != 0) {
            // FIXME: The C-side svm_model struct should be freed here.
            Cptr = 0;
        }
    }

    // C-side pointer to a svm_model.
    long Cptr;
}
