// Copyright (C) libsvm
// License: to be double checked.
package jcp.bindings.libsvm;

// NOTE: When using C libsvm instances of this class have to be reshaped in C
//       each time they are passed as arguments to libsvm methods.
//       Use a SparseDoubleMatrix2D for x and a double[] for y instead.
public class svm_problem
{
    public int l;
    public double[] y;
    public svm_node[][] x;

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
