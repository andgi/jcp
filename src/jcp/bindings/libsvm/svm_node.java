// Copyright (C) libsvm
// License: to be double checked.
package jcp.bindings.libsvm;

public class svm_node implements java.io.Serializable
{
    public int index;
    public double value;

    private static native void native_init();

    static {
        try {
            System.loadLibrary("svm-jni");
        } catch (UnsatisfiedLinkError e) {
            System.out.println
                ("Could not load native JNI wrapper code for libsvm.");
            System.exit(1);
        }
        native_init();
    }
}
