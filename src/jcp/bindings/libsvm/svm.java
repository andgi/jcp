// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.
package jcp.bindings.libsvm;
import java.io.*;
import java.util.*;
import java.lang.UnsupportedOperationException;

public class svm
{

    public static svm_model svm_train(svm_problem prob, svm_parameter param)
    {
        return new svm_model(native_svm_train(prob, param));
    }

    public static svm_model svm_train(svm_parameter param,
                                      SparseDoubleMatrix2D x,
                                      double[] y)
    {
        return new svm_model(native_svm_train_fast(param, x.Cptr, y));
    }

    public static void svm_cross_validation(svm_problem prob,
                                            svm_parameter param,
                                            int nr_fold,
                                            double[] target)
    {
        native_svm_cross_validation(prob, param, nr_fold, target);
    }

    public static int svm_get_svm_type(svm_model model)
    {
        return native_svm_get_svm_type(model.Cptr);
    }

    public static int svm_get_nr_class(svm_model model)
    {
        return native_svm_get_nr_class(model.Cptr);
    }

    public static void svm_get_labels(svm_model model, int[] label)
    {
        native_svm_get_labels(model.Cptr, label);
    }

    public static void svm_get_sv_indices(svm_model model, int[] indices)
    {
        native_svm_get_sv_indices(model.Cptr, indices);
    }

    public static int svm_get_nr_sv(svm_model model)
    {
        return native_svm_get_nr_sv(model.Cptr);
    }

    public static double svm_get_svr_probability(svm_model model)
    {
        return native_svm_get_svr_probability(model.Cptr);
    }

    public static double svm_predict_values(svm_model model,
                                            svm_node[] x,
                                            double[] dec_values)
    {
        return native_svm_predict_values(model.Cptr, x, dec_values);
    }

    public static double svm_predict(svm_model model, svm_node[] x)
    {
        return native_svm_predict(model.Cptr, x);
    }

    public static double svm_predict_probability(svm_model model,
                                                 svm_node[] x,
                                                 double[] prob_estimates)
    {
        return native_svm_predict_probability(model.Cptr, x, prob_estimates);
    }

    public static void svm_save_model(String model_file_name,
                                      svm_model model) throws IOException
    {
        int result = native_svm_save_model(model_file_name, model.Cptr);
        if (result == 0) {
            return;
        } else {
            throw new IOException("svm_save_model: Failed to save '" +
                                  model_file_name + "'.");
        }
    }

    public static svm_model svm_load_model(String model_file_name)
        throws IOException
    {
        long m = native_svm_load_model(model_file_name);
        if (m != 0) {
            return new svm_model(m);
        } else {
            throw new IOException("svm_load_model: Failed to load '" +
                                  model_file_name + "'.");
        }
    }

    public static svm_model svm_load_model(BufferedReader fp)
        throws IOException
    {
        // FIXME. If needed.
        throw new UnsupportedOperationException("Not implemented");
    }

    public static String svm_check_parameter(svm_problem prob,
                                             svm_parameter param)
    {
        return native_svm_check_parameter(prob, param);
    }

    public static String svm_check_parameter(svm_parameter param,
                                             SparseDoubleMatrix2D x,
                                             double[] y)
    {
        return native_svm_check_parameter_fast(param, x.Cptr, y);
    }

    public static int svm_check_probability_model(svm_model model)
    {
        return native_svm_check_probability_model(model.Cptr);
    }

    public static void svm_set_print_string_function
        (svm_print_interface print_func)
    {
        // FIXME. If needed.
        //throw new UnsupportedOperationException("Not implemented");
    }


    // Internal native functions.
    private static native long native_svm_train(svm_problem prob,
                                                svm_parameter param);
    private static native long native_svm_train_fast(svm_parameter param,
                                                     long x_ptr,
                                                     double[] y);
    private static native void native_svm_cross_validation(svm_problem prob,
                                                           svm_parameter param,
                                                           int nr_fold,
                                                           double[] target);
    private static native int native_svm_get_svm_type(long model_ptr);
    private static native int native_svm_get_nr_class(long model_ptr);
    private static native void native_svm_get_labels(long model_ptr,
                                                     int[] label);
    private static native void native_svm_get_sv_indices(long model_ptr,
                                                         int[] indices);
    private static native int native_svm_get_nr_sv(long model_ptr);
    private static native double native_svm_get_svr_probability(long model_ptr);
    private static native double native_svm_predict_values(long model_ptr,
                                                           svm_node[] x,
                                                           double[] dec_values);
    private static native double native_svm_predict(long model_ptr,
                                                    svm_node[] x);
    private static native double native_svm_predict_probability
        (long model_ptr,
         svm_node[] x,
         double[] prob_estimates);
    private static native int native_svm_save_model(String file_name,
                                                    long   model_ptr);
    private static native long native_svm_load_model(String file_name);
    private static native String native_svm_check_parameter
        (svm_problem prob,
         svm_parameter param);
    private static native String native_svm_check_parameter_fast
        (svm_parameter param,
         long x_ptr,
         double[] y);
    private static native int native_svm_check_probability_model
        (long model_ptr);

    static {
        try {
            System.loadLibrary("svm");
        } catch (UnsatisfiedLinkError e) {
            System.out.println
                ("Could not load libsvm.");
            System.exit(1);
        }
        try {
            System.loadLibrary("svm-jni");
        } catch (UnsatisfiedLinkError e) {
            System.out.println
                ("Could not load native JNI wrapper code for libsvm.");
            System.exit(1);
        }
    }
}
