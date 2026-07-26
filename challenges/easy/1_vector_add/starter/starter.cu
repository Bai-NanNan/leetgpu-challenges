#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    //计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //计算
    if (idx<N){
        C[idx] = A[idx] + B[idx];
    }
    //Benchmark   median 1.4218 ms  min 1.3957 ms  p20 1.4126 ms  p80 1.4381 ms
}

__global__ void vector_add_float4(const float* A, const float* B, float* C, int N) {
    //计算全局线程索引,不过每个Thread都要从4倍数开头取
    int idx = (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx*4 >= N) return;//假设是对齐的、padding过的
    //更宽的取值
    const float4* a4 = reinterpret_cast<const float4*>(A);
    const float4* b4 = reinterpret_cast<const float4*>(B);
    float4* c4 = reinterpret_cast<float4*>(C);
    float4 va = a4[idx];
    float4 vb = b4[idx];
    float4 vc;
    //计算
    vc.x = va.x + vb.x;
    vc.y = va.y + vb.y;
    vc.z = va.z + vb.z;
    vc.w = va.w + vb.w;
    c4[idx] = vc;
    //Benchmark   median 1.3903 ms  min 1.3499 ms  p20 1.3758 ms  p80 1.3982 ms
}



// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    //naive版本
    int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    //float4，每个thread处理四个
    int elementsPerThread = 4;
    int totalThreads = (N + elementsPerThread - 1) / elementsPerThread;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_float4<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);


    cudaDeviceSynchronize();
}
