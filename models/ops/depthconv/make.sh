cd src
nvcc -c -o depthconv_cuda_kernel.cu.o depthconv_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++14 -arch=sm_60
cd ..
CC=g++ python ./src/setup.py install
