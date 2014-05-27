// Copyright (C) 2014  Anders Gidenstam
// License: to be defined.

#include "jcp_bindings_libsvm_svm.h"
#include "jcp_bindings_libsvm_svm_node.h"
#include "jcp_bindings_libsvm_svm_problem.h"
#include "jcp_bindings_libsvm_svm_parameter.h"

#include <iostream>

#include <cstdlib>
#include <cstring>
#include <svm.h>

/* Internal functions. */ 
static struct svm_node* svm_node_array_from_java(JNIEnv* env,
                                                 jobjectArray jnodes);
static void free_svm_node_array(struct svm_node* nodes);

static struct svm_problem* svm_problem_from_java(JNIEnv* env,
                                                 jobject jproblem);
static void free_svm_problem(struct svm_problem* problem);

static struct svm_parameter* svm_parameter_from_java(JNIEnv* env,
                                                     jobject jparam);
static void free_svm_parameter(struct svm_parameter* param);

/* Internal shared data. */
/*   svm_node field IDs. */
jfieldID svm_node__index_FID;
jfieldID svm_node__value_FID;
/*   svm_problem field IDs. */
jfieldID svm_problem__l_FID;
jfieldID svm_problem__y_FID;
jfieldID svm_problem__x_FID;
/*   svm_parameter field IDs. */
jfieldID svm_parameter__svm_type_FID;
jfieldID svm_parameter__kernel_type_FID;
jfieldID svm_parameter__degree_FID;
jfieldID svm_parameter__gamma_FID;
jfieldID svm_parameter__coef0_FID;
jfieldID svm_parameter__cache_size_FID;
jfieldID svm_parameter__eps_FID;
jfieldID svm_parameter__C_FID;
jfieldID svm_parameter__nr_weight_FID;
jfieldID svm_parameter__weight_label_FID;
jfieldID svm_parameter__weight_FID;
jfieldID svm_parameter__nu_FID;
jfieldID svm_parameter__p_FID;
jfieldID svm_parameter__shrinking_FID;
jfieldID svm_parameter__probability_FID;

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_load_model
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1load_1model
    (JNIEnv* env,
     jclass  jsvm,
     jstring jfile_name)
{
    struct svm_model* model;
    const char* file_name = env->GetStringUTFChars(jfile_name, NULL);

    model = svm_load_model(file_name);

    env->ReleaseStringUTFChars(jfile_name, file_name);
    return (long)model;
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_save_model
 * Signature: (Ljava/lang/String;J)V
 */
JNIEXPORT jint JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1save_1model
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
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_train
 * Signature: (Ljcp/bindings/libsvm/svm_problem;Ljcp/bindings/libsvm/svm_parameter;)J
 */
JNIEXPORT jlong JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1train(JNIEnv* env,
                                                jclass  jsvm,
                                                jobject jproblem,
                                                jobject jparam)
{
    struct svm_problem* problem = svm_problem_from_java(env, jproblem);
    struct svm_parameter* param = svm_parameter_from_java(env, jparam);
    if (env->ExceptionOccurred()) {
        std::cerr << "Java_jcp_bindings_libsvm_svm_native_1svm_1train(): "
                  << "Java exception at argument conversion."
                  << std::endl;
        return 0;
    }

    struct svm_model* model = svm_train(problem, param);

    // The new svm_model reuses the support vector instance attributes
    // from the svm_problem struct.  Hence, these need to be compied
    // to a new memory area and the model set to free them,
    // eventually, if the svm_problem struct is to be freed now and
    // not leaked.
    // Copy the SV instances from the svm problem.
    for (int i = 0; i < model->l; i++) {
        struct svm_node* oldSV = model->SV[i];
        // Measure the length of the oldSV array.
        int len = 0;
        while (oldSV[len].index != -1) {
            len++;
        }
        len++;
        model->SV[i] =
            (struct svm_node*)malloc(len * sizeof(struct svm_node));
        memcpy(model->SV[i], oldSV, len * sizeof(struct svm_node));
    }
    model->free_sv = 1;

    free_svm_parameter(param);
    free_svm_problem(problem);

    return (long)model;
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_cross_validation
 * Signature: (Ljcp/bindings/libsvm/svm_problem;Ljcp/bindings/libsvm/svm_parameter;I[D)V
 */
JNIEXPORT void JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1cross_1validation
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
        std::cerr << "Java_jcp_bindings_libsvm_svm_native_1svm_1cross_1validation(): "
                  << "Java exception at argument conversion."
                  << std::endl;
        return;
    }

    svm_cross_validation(problem, param, jnr_fold, jtarget_elems);

    // FIXME: Verify that jtarget really is updated!
    env->ReleaseDoubleArrayElements(jtarget, jtarget_elems, JNI_COMMIT);
    free_svm_parameter(param);
    free_svm_problem(problem);
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_get_svm_type
 * Signature: (Ljcp/bindings/libsvm/svm_model;)I
 */
JNIEXPORT jint JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1get_1svm_1type
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_get_svm_type((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_get_nr_class
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1get_1nr_1class
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_get_nr_class((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_get_labels
 * Signature: (J[I)V
 */
JNIEXPORT void JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1get_1labels
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
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_get_sv_indices
 * Signature: (J[I)V
 */
JNIEXPORT void JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1get_1sv_1indices
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
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_get_nr_sv
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1get_1nr_1sv
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_get_nr_sv((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_get_svr_probability
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1get_1svr_1probability
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_get_svr_probability((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_predict_values
 * Signature: (J[Ljcp/bindings/libsvm/svm_node;[D)D
 */
JNIEXPORT jdouble JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1predict_1values
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
        std::cerr << "Java_jcp_bindings_libsvm_svm_native_1svm_1predict_1values(): "
                  << "Java exception at argument conversion."
                  << std::endl;
        return 0.0;
    }

    double result = svm_predict_values(model, instance, jdec_values_elems);

    // FIXME: Verify that jdec_values really is updated!
    env->ReleaseDoubleArrayElements(jdec_values, jdec_values_elems, JNI_COMMIT);
    free_svm_node_array(instance);

    return result;
}


/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_predict
 * Signature: (Ljcp/bindings/libsvm/svm_model;[Ljcp/bindings/libsvm/svm_node;)D
 */
JNIEXPORT jdouble JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1predict
    (JNIEnv*      env,
     jclass       jsvm,
     jlong        jmodel_ptr,
     jobjectArray jnodes)
{
    struct svm_node*  instance = svm_node_array_from_java(env, jnodes);
    struct svm_model* model = (struct svm_model*)jmodel_ptr;

    double result = svm_predict(model, instance);

    free_svm_node_array(instance);

    return result;
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_predict_probability
 * Signature: (J[Ljcp/bindings/libsvm/svm_node;[D)D
 */
JNIEXPORT jdouble JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1predict_1probability
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
        std::cerr << "Java_jcp_bindings_libsvm_svm_native_1svm_1predict_1probability():"
                  << " Java exception at argument conversion."
                  << std::endl;
        return 0.0;
    }

    double result = svm_predict_probability(model,
                                            instance,
                                            jprob_estimates_elems);

    // FIXME: Verify that jprob_estimates really is updated!
    env->ReleaseDoubleArrayElements(jprob_estimates,
                                    jprob_estimates_elems,
                                    JNI_COMMIT);
    free_svm_node_array(instance);

    return result; 
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_check_parameter
 * Signature: (Ljcp/bindings/libsvm/svm_problem;Ljcp/bindings/libsvm/svm_parameter;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1check_1parameter
    (JNIEnv* env,
     jclass  jsvm,
     jobject jproblem,
     jobject jparam)
{
    struct svm_problem* problem = svm_problem_from_java(env, jproblem);
    struct svm_parameter* param = svm_parameter_from_java(env, jparam);
    if (env->ExceptionOccurred()) {
        std::cerr << "Java_jcp_bindings_libsvm_svm_native_1svm_1check_1parameter(): Java exception."
                  << std::endl;
        return NULL;
    }

    const char* result = svm_check_parameter(problem, param);

    free_svm_parameter(param);
    free_svm_problem(problem);
    return env->NewStringUTF(result);
}

/*
 * Class:     jcp_bindings_libsvm_svm
 * Method:    native_svm_check_probability_model
 * Signature: (Ljcp/bindings/libsvm/svm_model;)I
 */
JNIEXPORT jint JNICALL
Java_jcp_bindings_libsvm_svm_native_1svm_1check_1probability_1model
    (JNIEnv* env,
     jclass  jsvm,
     jlong   jmodel_ptr)
{
    return svm_check_probability_model((const struct svm_model*)jmodel_ptr);
}

/*
 * Class:     jcp_bindings_libsvm_svm_node
 * Method:    native_init
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_jcp_bindings_libsvm_svm_1node_native_1init
    (JNIEnv* env,
     jclass  jsvm_node)
{
    svm_node__index_FID = env->GetFieldID(jsvm_node, "index", "I");
    svm_node__value_FID = env->GetFieldID(jsvm_node, "value", "D");
    if (env->ExceptionOccurred()) {
        std::cerr << "Java_jcp_bindings_libsvm_svm_1node_native_1init(): "
                  << "Java exception when caching field IDs."
                  << std::endl;
    }
}

/*
 * Class:     jcp_bindings_libsvm_svm_problem
 * Method:    native_init
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_jcp_bindings_libsvm_svm_1problem_native_1init
    (JNIEnv* env,
     jclass  jsvm_problem)
{
    svm_problem__l_FID = env->GetFieldID(jsvm_problem, "l", "I");
    svm_problem__y_FID = env->GetFieldID(jsvm_problem, "y", "[D");
    svm_problem__x_FID = env->GetFieldID(jsvm_problem,
                                         "x", "[[Ljcp/bindings/libsvm/svm_node;");
    if (env->ExceptionOccurred()) {
        std::cerr << "Java_jcp_bindings_libsvm_svm_1problem_native_1init(): "
                  << "Java exception when caching field IDs."
                  << std::endl;
    }
}

/*
 * Class:     jcp_bindings_libsvm_svm_parameter
 * Method:    native_init
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_jcp_bindings_libsvm_svm_1parameter_native_1init
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
    if (env->ExceptionOccurred()) {
        std::cerr << "Java_jcp_bindings_libsvm_svm_1parameter_native_1init(): "
                  << "Java exception when caching field IDs."
                  << std::endl;
    }
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
        result = (struct svm_node*)calloc(length + 1,
                                          sizeof(struct svm_node));

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
            std::cerr << "svm_node_array_from_java(): Java exception."
                      << std::endl;
            return NULL;
        }
    }

    return result;
}

static void free_svm_node_array(struct svm_node* nodes)
{
    free(nodes);
}

static struct svm_problem* svm_problem_from_java(JNIEnv* env,
                                                 jobject jproblem)
{
    struct svm_problem* problem =
        (struct svm_problem*)malloc(sizeof(struct svm_problem));

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

        problem->y = (double*)malloc(problem->l * sizeof(double));

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
            (struct svm_node**)malloc(problem->l * sizeof(struct svm_node*));

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
    free(problem->y);
    for (int i = 0; i < problem->l; i++) {
        free_svm_node_array(problem->x[i]);
    }
    free(problem->x);
    free(problem);
}

static struct svm_parameter* svm_parameter_from_java(JNIEnv* env,
                                                     jobject jparam)
{
    struct svm_parameter* param =
        (struct svm_parameter*)malloc(sizeof(struct svm_parameter));
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
    free(param);
}
