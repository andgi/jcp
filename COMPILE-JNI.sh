g++ -Wall -g -o lib/libsvm-jni.so jni/libsvm-jni.cpp -shared -fpic -I$JDKDIR/include/ -I$JDKDIR/include/linux/ -I../libsvm.git/ -L./lib -lsvm
