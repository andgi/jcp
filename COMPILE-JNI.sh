g++ -Wall -g -o lib/libsvm-jni.so jni/libsvm-jni.cpp -shared -fpic -I$JAVA_HOME/include/ -I$JAVA_HOME/include/linux/ -I../libsvm.git/ -L./lib -lsvm
