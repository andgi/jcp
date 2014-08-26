// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.libsvm;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class svm_model
    implements java.io.Serializable
{
    svm_model(long Cptr)
    {
        this.Cptr = Cptr;
    }

    protected void finalize()
        throws Throwable
    {
        if (Cptr != 0) {
            // FIXME: The C-side svm_model struct should be freed here.
            Cptr = 0;
        }
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        // Create a (likely) unique file name for the libsvm model.
        String fileName =
            Long.toHexString(Double.doubleToLongBits(Math.random())) +
            ".libsvm";

        // Write the libsvm model to a separate file.
        svm.svm_save_model(fileName, this);

        // Write the libsvm model file name to the Java output stream.
        oos.writeObject(fileName);
    }

    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        // Load libsvm model file name from the Java input stream.
        String fileName = (String)ois.readObject();

        // Load the C-side libsvm model from the designated file.
        this.Cptr = svm.svm_load_model(fileName).Cptr;
    }

    // C-side pointer to a svm_model.
    long Cptr;
}
