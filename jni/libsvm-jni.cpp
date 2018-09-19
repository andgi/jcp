// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2016, 2018  Anders Gidenstam
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

#include "se_hb_jcp_bindings_libsvm_svm.h"
#include "se_hb_jcp_bindings_libsvm_svm_model.h"
#include "se_hb_jcp_bindings_libsvm_svm_node.h"
#include "se_hb_jcp_bindings_libsvm_svm_problem.h"
#include "se_hb_jcp_bindings_libsvm_svm_parameter.h"
#include "se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D.h"
#include "se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D.h"
#include <iostream>

#include <cstdlib>
#include <cstring>
#include <svm.h>

#include <CERC/reference_counting>

/* Internal functions. */
static struct svm_node* svm_node_array_from_java(JNIEnv* env,
                                                 jobjectArray jnodes);

static struct svm_problem* svm_problem_from_java(JNIEnv* env,
                                                 jobject jproblem);
static void free_svm_problem(struct svm_problem* problem);

static struct svm_parameter* svm_parameter_from_java(JNIEnv* env,
                                                     jobject jparam);
static void free_svm_parameter(struct svm_parameter* param);
static void   compute_w(struct svm_model* model, double w[]);
static double compute_b(struct svm_model* model);
static void print_func(const char* str);

/* Internal shared data. */
/*   svm_node field IDs. */
static jfieldID svm_node__index_FID;
static jfieldID svm_node__value_FID;
/*   svm_problem field IDs. */
static jfieldID svm_problem__l_FID;
static jfieldID svm_problem__y_FID;
static jfieldID svm_problem__x_FID;
/*   svm_parameter field IDs. */
static jfieldID svm_parameter__svm_type_FID;
static jfieldID svm_parameter__kernel_type_FID;
static jfieldID svm_parameter__degree_FID;
static jfieldID svm_parameter__gamma_FID;
static jfieldID svm_parameter__coef0_FID;
static jfieldID svm_parameter__cache_size_FID;
static jfieldID svm_parameter__eps_FID;
static jfieldID svm_parameter__C_FID;
static jfieldID svm_parameter__nr_weight_FID;
static jfieldID svm_parameter__weight_label_FID;
static jfieldID svm_parameter__weight_FID;
static jfieldID svm_parameter__nu_FID;
static jfieldID svm_parameter__p_FID;
static jfieldID svm_parameter__shrinking_FID;
static jfieldID svm_parameter__probability_FID;

/* Reference counters for svm_node arrays. */
static cerc::reference_counting<svm_node> instance_rc;

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
/* libsvm functions. */

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_load_model
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1load_1model
    (JNIEnv* env,
     jclass  jsvm,
     jstring jfile_name)
{
    struct svm_model* model;
    const char* file_name = env->GetStringUTFChars(jfile_name, NULL);

    model = svm_load_model(file_name);
    // Increase the RC of the support vector instance attributes and set
    // the model to not automatically free these.
    // FIXME: This seems to cause segmentation faults and, besides,
    //        there is currently no mechanism for gaining Java-side
    //        access to the SVs.
    /*for (int i = 0; i < model->l; i++) {
        instance_rc.inc(model->SV[i]);
    }
    model->free_sv = 0;*/

    env->ReleaseStringUTFChars(jfile_name, file_name);
    return (long)model;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_save_model
 * Signature: (Ljava/lang/String;J)V
 */
JNIEXPORT jint JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1save_1model
    (JNIEnv* env,
     jclass  jsvm,
     jstring jfile_name,
     jlong   jmodel_ptr)
{
    int res;
    const char* file_name = env->GetStringUTFChars(jfile_name, NULL);

    res = svm_save_model(file_name, (const struct svm_model*)jmodel_ptr);

    env->ReleaseStringUTFChars(jfile_name, file_name);
    return res;
}


/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_train
 * Signature: (Ljcp/bindings/libsvm/svm_problem;Ljcp/bindings/libsvm/svm_parameter;)J
 */
JNIEXPORT jlong JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1train
    (JNIEnv* env,
     jclass  jsvm,
     jobject jproblem,
     jobject jparam)
{
    struct svm_problem* problem = svm_problem_from_java(env, jproblem);
    struct svm_parameter* param = svm_parameter_from_java(env, jparam);
    if (env->ExceptionOccurred()) {
        std::cerr << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1train(): "
                  << "Java exception at argument conversion."
                  << std::endl;
        return 0;
    }

    struct svm_model* model = svm_train(problem, param);

    // The new svm_model reuses the support vector instance attributes
    // from the svm_problem struct.  Hence, the reference count of these need
    // to be increased.
    // Increase the RC of the supporting SV instances from the svm problem.
    for (int i = 0; i < model->l; i++) {
        instance_rc.inc(model->SV[i]);
    }
    free_svm_parameter(param);
    free_svm_problem(problem);

    return (long)model;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_train_fast
 * Signature: (Ljcp/bindings/libsvm/svm_parameter;J[D)J
 */
JNIEXPORT jlong JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1train_1fast
   (JNIEnv*      env,
    jclass       jsvm,
    jobject      jparam,
    jlong        jx_ptr,
    jdoubleArray jy)
{
    struct svm_parameter* param = svm_parameter_from_java(env, jparam);
    struct svm_problem problem;
    /* Set up the svm_problem struct. */
    // NOTE:
    //   The new svm_model reuses the support vector instance attributes
    //   from x in the svm_problem struct.  Hence, the storage for
    //   problem->x (jx_ptr) must not be freed while the model remains.
    problem.l = env->GetArrayLength(jy);
    problem.x = (struct svm_node **)jx_ptr;
    problem.y = env->GetDoubleArrayElements(jy, NULL);

    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1train_1fast(): "
            << "Java exception at argument conversion."
            << std::endl;
        return 0;
    }

    struct svm_model* model = svm_train(&problem, param);
    // Increase the RC of the supporting SV instances from the svm problem.
    for (int i = 0; i < model->l; i++) {
        instance_rc.inc(model->SV[i]);
    }

    env->ReleaseDoubleArrayElements(jy, problem.y, JNI_ABORT);
    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1train_1fast(): "
            << "Java exception." << std::endl;
        return 0;
    }
    problem.y = NULL;
    free_svm_parameter(param);

    return (long)model;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_cross_validation
 * Signature: (Ljcp/bindings/libsvm/svm_problem;Ljcp/bindings/libsvm/svm_parameter;I[D)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1cross_1validation
    (JNIEnv*      env,
     jclass       jsvm,
     jobject      jproblem,
     jobject      jparam,
     jint         jnr_fold,
     jdoubleArray jtarget)
{
    struct svm_problem* problem = svm_problem_from_java(env, jproblem);
    struct svm_parameter* param = svm_parameter_from_java(env, jparam);
    jdouble* jtarget_elems      = env->GetDoubleArrayElements(jtarget, NULL);
    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1cross_1validation(): "
            << "Java exception at argument conversion."
            << std::endl;
        return;
    }

    // FIXME: What happens with the RC of the instances? Should be no problem.
    svm_cross_validation(problem, param, jnr_fold, jtarget_elems);

    // FIXME: Verify that jtarget really is updated!
    env->ReleaseDoubleArrayElements(jtarget, jtarget_elems, JNI_COMMIT);
    free_svm_parameter(param);
    free_svm_problem(problem);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_get_svm_type
 * Signature: (Ljcp/bindings/libsvm/svm_model;)I
 */
JNIEXPORT jint JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1get_1svm_1type
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_get_svm_type((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_get_nr_class
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1get_1nr_1class
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_get_nr_class((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_get_labels
 * Signature: (J[I)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1get_1labels
    (JNIEnv*   env,
     jclass    jsvm,
     jlong     jmodel_ptr,
     jintArray jlabels)
{
    int* labels = env->GetIntArrayElements(jlabels, NULL);

    svm_get_labels((const struct svm_model*)jmodel_ptr, labels);

    env->ReleaseIntArrayElements(jlabels, labels, JNI_COMMIT);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_get_sv_indices
 * Signature: (J[I)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1get_1sv_1indices
    (JNIEnv*   env,
     jclass    jsvm,
     jlong     jmodel_ptr,
     jintArray jindices)
{
    int* indices = env->GetIntArrayElements(jindices, NULL);

    svm_get_sv_indices((const struct svm_model*)jmodel_ptr, indices);

    env->ReleaseIntArrayElements(jindices, indices, JNI_COMMIT);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_get_nr_sv
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1get_1nr_1sv
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_get_nr_sv((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_get_svr_probability
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1get_1svr_1probability
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_get_svr_probability((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_predict_values
 * Signature: (J[Ljcp/bindings/libsvm/svm_node;[D)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict_1values
    (JNIEnv*       env,
     jclass        jsvm,
     jlong         jmodel_ptr,
     jobjectArray  jnodes,
     jdoubleArray  jdec_values)
{
    struct svm_node*  instance = svm_node_array_from_java(env, jnodes);
    struct svm_model* model = (struct svm_model*)jmodel_ptr;
    jdouble* jdec_values_elems = env->GetDoubleArrayElements(jdec_values, NULL);
    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict_1values(): "
            << "Java exception at argument conversion."
            << std::endl;
        return 0.0;
    }

    double result = svm_predict_values(model, instance, jdec_values_elems);

    // FIXME: Verify that jdec_values really is updated!
    env->ReleaseDoubleArrayElements(jdec_values, jdec_values_elems, JNI_COMMIT);
    instance_rc.dec(instance);

    return result;
}


/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_predict
 * Signature: (Ljcp/bindings/libsvm/svm_model;[Ljcp/bindings/libsvm/svm_node;)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict
    (JNIEnv*      env,
     jclass       jsvm,
     jlong        jmodel_ptr,
     jobjectArray jnodes)
{
    struct svm_node*  instance = svm_node_array_from_java(env, jnodes);
    struct svm_model* model = (struct svm_model*)jmodel_ptr;

    double result = svm_predict(model, instance);

    instance_rc.dec(instance);

    return result;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_predict_fast
 * Signature: (JJ)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict_1fast
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr,
     jlong   jinstance_ptr)
{
    struct svm_model* model = (struct svm_model*)jmodel_ptr;
    struct svm_node * instance = *(struct svm_node **)jinstance_ptr;
    if (env->ExceptionOccurred()) {
      std::cerr
          << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict_1fast():"
          << " Java exception at argument conversion."
          << std::endl;
        return 0.0;
    }
    double result = svm_predict(model,
                                instance);

    return result;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_predict_probability
 * Signature: (J[Ljcp/bindings/libsvm/svm_node;[D)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict_1probability
    (JNIEnv*      env,
     jclass       jsvm,
     jlong        jmodel_ptr,
     jobjectArray jnodes,
     jdoubleArray jprob_estimates)
{
    struct svm_node*  instance = svm_node_array_from_java(env, jnodes);
    struct svm_model* model = (struct svm_model*)jmodel_ptr;
    jdouble* jprob_estimates_elems =
        env->GetDoubleArrayElements(jprob_estimates, NULL);
    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict_1probability():"
            << " Java exception at argument conversion."
            << std::endl;
        return 0.0;
    }

    double result = svm_predict_probability(model,
                                            instance,
                                            jprob_estimates_elems);

    env->ReleaseDoubleArrayElements(jprob_estimates,
                                    jprob_estimates_elems,
                                    JNI_COMMIT);
    instance_rc.dec(instance);

    return result; 
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_predict_probability_fast
 * Signature: (JJ[D)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict_1probability_1fast
    (JNIEnv*      env,
     jclass       jsvm,
     jlong        jmodel_ptr,
     jlong        jinstance_ptr,
     jdoubleArray jprob_estimates)
{
    struct svm_model* model = (struct svm_model*)jmodel_ptr;
    struct svm_node * instance = *(struct svm_node **)jinstance_ptr;
    jdouble* jprob_estimates_elems =
        env->GetDoubleArrayElements(jprob_estimates, NULL);
    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1predict_1probability_1fast():"
            << " Java exception at argument conversion."
            << std::endl;
        return 0.0;
    }
    double result = svm_predict_probability(model,
                                            instance,
                                            jprob_estimates_elems);

    env->ReleaseDoubleArrayElements(jprob_estimates,
                                    jprob_estimates_elems,
                                    JNI_COMMIT);
    return result;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_check_parameter
 * Signature: (Ljcp/bindings/libsvm/svm_problem;Ljcp/bindings/libsvm/svm_parameter;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1check_1parameter
    (JNIEnv* env,
     jclass  jsvm,
     jobject jproblem,
     jobject jparam)
{
    struct svm_problem* problem = svm_problem_from_java(env, jproblem);
    struct svm_parameter* param = svm_parameter_from_java(env, jparam);
    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1check_1parameter(): Java exception."
            << std::endl;
        return NULL;
    }

    const char* result = svm_check_parameter(problem, param);

    free_svm_parameter(param);
    free_svm_problem(problem);
    return env->NewStringUTF(result);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_check_parameter_fast
 * Signature: (Ljcp/bindings/libsvm/svm_parameter;J[D)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1check_1parameter_1fast
    (JNIEnv*      env,
     jclass       jsvm,
     jobject      jparam,
     jlong        jx_ptr,
     jdoubleArray jy)
{
    struct svm_parameter* param = svm_parameter_from_java(env, jparam);
    struct svm_problem problem;
    /* Set up the svm_problem struct. */
    problem.l = env->GetArrayLength(jy);
    problem.x = (struct svm_node **)jx_ptr;
    problem.y = env->GetDoubleArrayElements(jy, NULL);
    if (env->ExceptionOccurred()) {
        std::cerr << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1check_1parameter_fast(): Java exception."
                  << std::endl;
        return NULL;
    }

    const char* result = svm_check_parameter(&problem, param);

    env->ReleaseDoubleArrayElements(jy, problem.y, JNI_ABORT);
    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1check_1parameter_1fast(): "
            << "Java exception." << std::endl;
        return 0;
    }
    problem.y = NULL;
    free_svm_parameter(param);
    return env->NewStringUTF(result);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_check_probability_model
 * Signature: (Ljcp/bindings/libsvm/svm_model;)I
 */
JNIEXPORT jint JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1check_1probability_1model
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_check_probability_model((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm_model
 * Method:    native_free_svm_model
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_1model_native_1free_1svm_1model
    (JNIEnv* env,
     jclass  jsvm_model,
     jlong   jmodel_ptr)
{
    struct svm_model* model = (struct svm_model*)jmodel_ptr;
#ifdef DEBUG
    std::cerr
        << "Java_se_hb_jcp_bindings_libsvm_svm_1model"
        << "_native_1free_1svm_1model(): "
        << "Freeing svm_model at " << model << "." << std::endl;
#endif
    if (model) {
        if (!model->free_sv) {
            // Decrease the RC of the shared supporting SV instances from the
            // svm problem.
            for (int i = 0; i < model->l; i++) {
                instance_rc.dec(model->SV[i]);
            }
            svm_free_and_destroy_model(&model);
        } else {
            // FIXME: This seems to cause segmentation faults for models
            //        loaded via svm_load_model(), even without any tampering
            //        here in libsvm-jni. For now, the model is leaked.
            //svm_free_and_destroy_model(&model);
        }
    }
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm
 * Method:    native_svm_distance_from_separating_plane
 * Signature: (JJJ)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_native_1svm_1distance_1from_1separating_1plane
    (JNIEnv*      env,
     jclass       jsvm,
     jlong        jmodel_ptr,
     jlong        jinstance_ptr,
     jlong        attribute_count)
{
    struct svm_model* model = (struct svm_model*)jmodel_ptr;
    struct svm_node*  instance = *(struct svm_node **)jinstance_ptr;

    // FIXME: This is only valid for 1-class SVM and 2-class SVM with
    //        classes -1.0 and 1.0.
    // See the libSVM FAQ #804,
    // http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f804 for the internal
    // details.
    double* w = (double*)std::calloc(sizeof(double), attribute_count);
    compute_w(model, w);
    double distance = compute_b(model);

    if (model->nr_class == 2) {
        for (int i = 0; instance[i].index >= 0; i++) {
            distance += w[instance[i].index] * instance[i].value;
        }
    } else {
        // Not implemented.
    }
    std::free(w);
    return distance;
}

/******************************************************************************/
/* libsvm Java interface class initialization functions. */

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm_node
 * Method:    native_init
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_1node_native_1init
    (JNIEnv* env,
     jclass  jsvm_node)
{
    svm_node__index_FID = env->GetFieldID(jsvm_node, "index", "I");
    svm_node__value_FID = env->GetFieldID(jsvm_node, "value", "D");
    if (env->ExceptionOccurred()) {
        std::cerr << "Java_se_hb_jcp_bindings_libsvm_svm_1node_native_1init(): "
                  << "Java exception when caching field IDs."
                  << std::endl;
    }
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm_problem
 * Method:    native_init
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_1problem_native_1init
    (JNIEnv* env,
     jclass  jsvm_problem)
{
    svm_problem__l_FID = env->GetFieldID(jsvm_problem, "l", "I");
    svm_problem__y_FID = env->GetFieldID(jsvm_problem, "y", "[D");
    svm_problem__x_FID = env->GetFieldID(jsvm_problem,
                                         "x", "[[Ljcp/bindings/libsvm/svm_node;");
    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_1problem_native_1init(): "
            << "Java exception when caching field IDs."
            << std::endl;
    }
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_svm_parameter
 * Method:    native_init
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_svm_1parameter_native_1init
    (JNIEnv* env,
     jclass  jsvm_parameter)
{
    svm_parameter__svm_type_FID = env->GetFieldID(jsvm_parameter,
                                                  "svm_type", "I");
    svm_parameter__kernel_type_FID = env->GetFieldID(jsvm_parameter,
                                                     "kernel_type", "I");
    svm_parameter__degree_FID = env->GetFieldID(jsvm_parameter,
                                                "degree", "I");
    svm_parameter__gamma_FID = env->GetFieldID(jsvm_parameter, "gamma", "D");
    svm_parameter__coef0_FID = env->GetFieldID(jsvm_parameter, "coef0", "D");
    svm_parameter__cache_size_FID = env->GetFieldID(jsvm_parameter,
                                                    "cache_size", "D");
    svm_parameter__eps_FID = env->GetFieldID(jsvm_parameter, "eps", "D");
    svm_parameter__C_FID = env->GetFieldID(jsvm_parameter, "C", "D");
    svm_parameter__nr_weight_FID = env->GetFieldID(jsvm_parameter,
                                                   "nr_weight", "I");
    svm_parameter__weight_label_FID = env->GetFieldID(jsvm_parameter,
                                                      "weight_label", "[I");
    svm_parameter__weight_FID = env->GetFieldID(jsvm_parameter, "weight", "[D");
    svm_parameter__nu_FID = env->GetFieldID(jsvm_parameter, "nu", "D");
    svm_parameter__p_FID = env->GetFieldID(jsvm_parameter, "p", "D");
    svm_parameter__shrinking_FID = env->GetFieldID(jsvm_parameter,
                                                   "shrinking", "I");
    svm_parameter__probability_FID = env->GetFieldID(jsvm_parameter,
                                                     "probability", "I");

    svm_set_print_string_function(print_func);

    if (env->ExceptionOccurred()) {
        std::cerr
            << "Java_se_hb_jcp_bindings_libsvm_svm_1parameter_native_1init(): "
            << "Java exception when caching field IDs."
            << std::endl;
    }
}

/******************************************************************************/
/* libsvm storage functions. */

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D
 * Method:    native_vector_create
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1create
    (JNIEnv* env,
     jclass  jSDM1D,
     jint    size)
{
    // Allocate one svm_node and mark it as EOL.
    struct svm_node** vptr =
        (struct svm_node**)std::malloc(sizeof(struct svm_node*));
    *vptr = (struct svm_node*)std::malloc(1 * sizeof(struct svm_node));
    instance_rc.inc(*vptr);
    (*vptr)[0].index = -1;
    return (jlong)vptr;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D
 * Method:    native_vector_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1free
    (JNIEnv*  env,
     jclass   jSDM1D,
     jlong    jptr)
{
    struct svm_node** vptr = (struct svm_node**)jptr;
    instance_rc.dec(*vptr);
    free(vptr);
#ifdef DEBUG
    std::cerr << "Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1free(): "
              << "Freed vector at " << (jlong)vptr << " -> "
              << (jlong)*(struct svm_node**)jptr
              << "." << std::endl;
#endif
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D
 * Method:    native_vector_create_from
 * Signature: (I[I[D)J
 */
JNIEXPORT jlong JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1create_1from
    (JNIEnv*      env,
     jclass       jSDM1D,
     jintArray    jindices,
     jdoubleArray jvalues)
{
    int i;
    jint*    indices = env->GetIntArrayElements(jindices, NULL);
    jdouble* values = env->GetDoubleArrayElements(jvalues, NULL);
    int length = env->GetArrayLength(jvalues);
    struct svm_node** vptr =
        (struct svm_node**)std::malloc(sizeof(struct svm_node*));
    *vptr =
        (struct svm_node*)std::malloc((length + 1) * sizeof(struct svm_node));
    instance_rc.inc(*vptr);

    for (i = 0; i < length; i++) {
        (*vptr)[i].index = indices[i];
        (*vptr)[i].value = values[i];
        //printf(" %d: (%d, %e)\n", i, (*vptr)[i].index, (*vptr)v[i].value);
    }
    (*vptr)[i].index = -1;
    (*vptr)[i].value = 0.0;

    env->ReleaseDoubleArrayElements(jvalues, values, JNI_ABORT);
    env->ReleaseIntArrayElements(jindices, indices, JNI_ABORT);
    if (env->ExceptionOccurred()) {
        // FIXME: May leak the svm_node array.
        std::cerr << "Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1create_1from(): "
                  << "Java exception."
                  << std::endl;
        return 0;
    }
    return (jlong)vptr;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D
 * Method:    native_vector_assign
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1assign
    (JNIEnv* env,
     jclass  jSDM1D,
     jlong   dest_jptr,
     jlong   src_jptr)
{
    struct svm_node** dest_vptr = (struct svm_node**)dest_jptr;
    struct svm_node** src_vptr  = (struct svm_node**)src_jptr;

    if (*dest_vptr != *src_vptr) {
        instance_rc.dec(*dest_vptr);
        *dest_vptr = *src_vptr;
        instance_rc.inc(*dest_vptr);
    }
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D
 * Method:    native_vector_get
 * Signature: (JI)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1get
    (JNIEnv* env,
     jclass  jSDM1D,
     jlong   jptr,
     jint    index)
{
    struct svm_node** vptr = (struct svm_node**)jptr;
    int i = 0;
    while ((*vptr)[i].index != -1 && (*vptr)[i].index < index) {
        i++;
    }
    return ((*vptr)[i].index == index) ? (*vptr)[i].value : 0.0;

}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D
 * Method:    native_vector_set
 * Signature: (JID)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1set
    (JNIEnv* env,
     jclass  jSDM1D,
     jlong   jptr,
     jint    index,
     jdouble value)
{
    struct svm_node** vptr = (struct svm_node**)jptr;
    int i = 0;
    while ((*vptr)[i].index != -1) {
        if ((*vptr)[i].index == index) {
            (*vptr)[i].value = value;
            return;
        }
        i++;
    }

    // The requested element was not found.
    if (value == 0.0) {
        // Done if the value to set is zero.
        return;
    }
    // Allocate a new larger array and copy the contents and the new element.
    // FIXME: Using set() to initialize a matrix row element by element
    //        will be slow.
    struct svm_node* old = *vptr;
    *vptr = (struct svm_node*)std::malloc((i+2) * sizeof(struct svm_node));
    instance_rc.inc(*vptr);
    i = 0;
    while (old[i].index != -1 && old[i].index < index) {
        (*vptr)[i].index = old[i].index;
        (*vptr)[i].value = old[i].value;
        i++;
    }
    (*vptr)[i].index = index;
    (*vptr)[i].value = value;
    while (old[i].index != -1) {
        (*vptr)[i+1].index = old[i].index;
        (*vptr)[i+1].value = old[i].value;
        i++;
    }
    (*vptr)[i+1].index = -1;
    instance_rc.dec(old);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D
 * Method:    native_matrix_create
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D_native_1matrix_1create
    (JNIEnv* env,
     jclass  jSDM2D,
     jint    rows,
     jint    columns)
{
    struct svm_node** m =
        (struct svm_node**)std::malloc(rows * sizeof(struct svm_node*));
    for (int i = 0; i < rows; i++) {
        // Allocate one svm_node for each row.
        m[i] = (struct svm_node*)std::malloc(1 * sizeof(struct svm_node));
        instance_rc.inc(m[i]);
        // Mark it as EOL.
        m[i][0].index = -1;
    }
#ifdef DEBUG
    std::cerr << "Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D_native_1matrix_1create(): "
              << "Created " << rows << " x " << columns << " matrix at "
              << (jlong)m << "." << std::endl;
#endif
    return (jlong)m;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D
 * Method:    native_matrix_free
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D_native_1matrix_1free
    (JNIEnv* env,
     jclass  jSDM2D,
     jlong   jptr,
     jint    rows)
{
    struct svm_node** m = (struct svm_node**)jptr;

    for (int i = 0; i < rows; i++) {
        instance_rc.dec(m[i]);
    }
    std::free(m);
#ifdef DEBUG
    std::cerr << "Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D_native_1matrix_1free(): "
              << "Freed matrix at " << (jlong)m << "." << std::endl;
#endif
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D
 * Method:    native_matrix_get
 * Signature: (JII)D
 */
JNIEXPORT jdouble JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D_native_1matrix_1get
    (JNIEnv* env,
     jclass  jSDM2D,
     jlong   jptr,
     jint    row,
     jint    column)
{
    struct svm_node** m = (struct svm_node**)jptr;
    int i = 0;
    while (m[row][i].index != -1 && m[row][i].index < column) {
        i++;
    }
    return (m[row][i].index == column) ? m[row][i].value : 0.0;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D
 * Method:    native_matrix_set
 * Signature: (JIID)V
 */
JNIEXPORT void JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D_native_1matrix_1set
    (JNIEnv* env,
     jclass  jSDM2D,
     jlong   jptr,
     jint    row,
     jint    column,
     jdouble value)
{
    struct svm_node** m = (struct svm_node**)jptr;
    int i = 0;
    while (m[row][i].index != -1) {
        if (m[row][i].index == column) {
            m[row][i].value = value;
            return;
        }
        i++;
    }
    // The requested element was not found.
    // Allocate a new larger array and copy the contents and the new element.
    // FIXME: Using set() to initialize a matrix row element by element
    //        will be slow.
    struct svm_node* old = m[row];
    m[row] = (struct svm_node*)std::malloc((i+2) * sizeof(struct svm_node));
    instance_rc.inc(m[row]);
    i = 0;
    while (old[i].index != -1 && old[i].index < column) {
        m[row][i].index = old[i].index;
        m[row][i].value = old[i].value;
        i++;
    }
    m[row][i].index = column;
    m[row][i].value = value;
    while (old[i].index != -1) {
        m[row][i+1].index = old[i].index;
        m[row][i+1].value = old[i].value;
        i++;
    }
    m[row][i+1].index = -1;
    instance_rc.dec(old);
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D
 * Method:    native_matrix_get_row
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D_native_1matrix_1get_1row
    (JNIEnv* env,
     jclass  jSDM2D,
     jlong   jptr,
     jint    row)
{
    struct svm_node** vptr = &((struct svm_node**)jptr)[row];
    // NOTE: The row view is just a view and should not increase the RC
    //       as it already ensures that its parent matrix is kept alive.
    //instance_rc.inc(*vptr);
    return (jlong)vptr;
}

/*
 * Class:     se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D
 * Method:    native_matrix_set_row
 * Signature: (JI[I[D)J
 */
JNIEXPORT jlong JNICALL
Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix2D_native_1matrix_1set_1row
    (JNIEnv*      env,
     jclass       jSDM2D,
     jlong        jptr,
     jint         row,
     jintArray    jcolumns,
     jdoubleArray jvalues)
{
    struct svm_node** m = (struct svm_node**)jptr;
    instance_rc.dec(m[row]);
    struct svm_node** v =
        (struct svm_node**)Java_se_hb_jcp_bindings_libsvm_SparseDoubleMatrix1D_native_1vector_1create_1from
            (env, jSDM2D, jcolumns, jvalues);
    m[row] = *v; // Note: the RC of the new m[row] is already 1.
    std::free(v);

    return (jlong)&m[row];
}

#ifdef __cplusplus
}
#endif

/******************************************************************************/
/* Internal functions. */

static struct svm_node* svm_node_array_from_java(JNIEnv* env,
                                                 jobjectArray jnodes)
{
    int i;
    int length = env->GetArrayLength(jnodes);
    struct svm_node* result = NULL;

    if (length > 0) {
        result =
            (struct svm_node*)std::malloc((length + 1) * sizeof(struct svm_node));
        instance_rc.inc(result);

        for (i = 0; i < length; i++) {
            jobject elem = env->GetObjectArrayElement(jnodes, i);

            result[i].index = env->GetIntField(elem, svm_node__index_FID);
            result[i].value = env->GetDoubleField(elem, svm_node__value_FID);
            //printf(" %d: (%d, %e)\n", i, result[i].index, result[i].value);

            env->DeleteLocalRef(elem);
        }
        result[i].index = -1;
        result[i].value = 0.0;

        if (env->ExceptionOccurred()) {
            // FIXME: Might leak the svm_node array.
            std::cerr << "svm_node_array_from_java(): Java exception."
                      << std::endl;
            return NULL;
        }
    }

    return result;
}

static struct svm_problem* svm_problem_from_java(JNIEnv* env,
                                                 jobject jproblem)
{
    struct svm_problem* problem =
        (struct svm_problem*)std::malloc(sizeof(struct svm_problem));

    /* Copy problem->l value. */
    problem->l = env->GetIntField(jproblem, svm_problem__l_FID);
    if (env->ExceptionOccurred()) {
        std::cerr << "svm_problem_from_java(): Java exception."
                  << std::endl;
        return NULL;
    }

    /* Copy the target values (problem->y). */
    {
        jdoubleArray jy =
            (jdoubleArray)env->GetObjectField(jproblem, svm_problem__y_FID);
        jdouble* jy_elems = env->GetDoubleArrayElements(jy, NULL);

        problem->y = (double*)std::malloc(problem->l * sizeof(double));

        for (int i = 0; i < problem->l; i++) {
            problem->y[i] = jy_elems[i];
        }

        env->ReleaseDoubleArrayElements(jy, jy_elems, JNI_ABORT);
    }
    if (env->ExceptionOccurred()) {
        std::cerr << "svm_problem_from_java(): Java exception."
                  << std::endl;
        return NULL;
    }

    /* Copy the instance attributes (problem->x). */
    {
        jobjectArray jx =
            (jobjectArray)env->GetObjectField(jproblem, svm_problem__x_FID);

        problem->x =
            (struct svm_node**)std::malloc(problem->l * sizeof(struct svm_node*));

        for (int i = 0; i < problem->l; i++) {
            jobjectArray jinstance =
                (jobjectArray)env->GetObjectArrayElement(jx, i);
            if (env->ExceptionOccurred()) {
                std::cerr << "svm_problem_from_java(): "
                          << "Java exception reading problem->x[" << i << "]."
                          << std::endl;
                return NULL;
            }

            problem->x[i] = svm_node_array_from_java(env, jinstance);
            env->DeleteLocalRef(jinstance);
        }
    }
    if (env->ExceptionOccurred()) {
        std::cerr << "svm_problem_from_java(): Java exception."
                  << std::endl;
        return NULL;
    }

    return problem;
}

static void free_svm_problem(struct svm_problem* problem)
{
    std::free(problem->y);
    for (int i = 0; i < problem->l; i++) {
        instance_rc.dec(problem->x[i]);
    }
    std::free(problem->x);
    std::free(problem);
}

static struct svm_parameter* svm_parameter_from_java(JNIEnv* env,
                                                     jobject jparam)
{
    struct svm_parameter* param =
        (struct svm_parameter*)std::malloc(sizeof(struct svm_parameter));
    /* Copy param->svm_type value. */
    param->svm_type = env->GetIntField(jparam, svm_parameter__svm_type_FID);
    /* Copy param->kernel_type value. */
    param->kernel_type =
        env->GetIntField(jparam, svm_parameter__kernel_type_FID);
    /* Copy param->degree value. */
    param->degree = env->GetIntField(jparam, svm_parameter__degree_FID);
    /* Copy param->gamma value. */
    param->gamma = env->GetDoubleField(jparam, svm_parameter__gamma_FID);
    /* Copy param->coef0 value. */
    param->coef0 = env->GetDoubleField(jparam, svm_parameter__coef0_FID);
    /* Copy param->cache_size value. */
    param->cache_size =
        env->GetDoubleField(jparam, svm_parameter__cache_size_FID);
    /* Copy param->eps value. */
    param->eps = env->GetDoubleField(jparam, svm_parameter__eps_FID);
    /* Copy param->C value. */
    param->C = env->GetDoubleField(jparam, svm_parameter__C_FID);
    /* Copy param->nr_weight value. */
    param->nr_weight = env->GetIntField(jparam, svm_parameter__nr_weight_FID);
    /* FIXME: Temporary set to 0. */
    param->weight_label = NULL;
    param->weight = NULL;
    /* Copy param->nu value. */
    param->nu = env->GetDoubleField(jparam, svm_parameter__nu_FID);
    /* Copy param->p value. */
    param->p = env->GetDoubleField(jparam, svm_parameter__p_FID);
    /* Copy param->shrinking value. */
    param->shrinking = env->GetIntField(jparam, svm_parameter__shrinking_FID);
    /* Copy param->probability value. */
    param->probability =
        env->GetIntField(jparam, svm_parameter__probability_FID);
    if (env->ExceptionOccurred()) {
        std::cerr << "svm_parameter_from_java(): Java exception."
                  << std::endl;
        return NULL;
    }
    
    return param;
}

static void free_svm_parameter(struct svm_parameter* param)
{
    std::free(param);
}

static void compute_w(struct svm_model* model, double w[])
{
    // FIXME: This is only valid for 1-class SVM and 2-class SVM with
    //        classes -1.0 and 1.0.
    double sign = (model->label != 0 && model->label[0] == -1.0)
                  ? -1.0 : 1.0;
    for (int l = 0; l < model->l; l++) {
        for (int i = 0; model->SV[l][i].index >= 0; i++) {
            w[model->SV[l][i].index] +=
                sign * model->sv_coef[0][l] * model->SV[l][i].value;
        }
    }
}

static double compute_b(struct svm_model* model)
{
    // FIXME: This is only valid for 1-class SVM and 2-class SVM with
    //        classes -1.0 and 1.0.
    double sign = (model->label != 0 && model->label[0] == -1.0)
                  ? -1.0 : 1.0;
    return sign * -model->rho[0];
}

static void print_func(const char* str)
{
}
