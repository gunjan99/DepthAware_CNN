cd src
nvcc -c -o depthavgpooling_cuda_kernel.cu.o depthavgpooling_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++14 -arch=sm_60
cd ..
CC=g++ python3 ./src/setup.py install
