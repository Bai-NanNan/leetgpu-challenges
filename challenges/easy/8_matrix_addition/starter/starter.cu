#include <cuda_runtime.h>

__global__ void matrix_add(const float* __restrict__ A, const float* __restrict__ B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (index < total) {
        C[index] = A[index] + B[index];
    }
    //Benchmark   median 1.1787 ms  min 1.1557 ms  p20 1.1708 ms  p80 1.2034 ms
}

//固定尺寸的Grid循环完成计算，且float4访存
__global__ void matrix_add_GridStridLoop_float4(
    const float4* __restrict__ A, 
    const float4* __restrict__ B, 
    float4* C, 
    int vector_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;//每步跨越Thread总数
    for(;index<vector_count;index += stride){
        float4 a = A[index];
        float4 b = B[index];

        C[index] = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
    }
    //Benchmark   median 0.9398 ms  min 0.9272 ms  p20 0.9313 ms  p80 0.9573 ms

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    // matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    int total = N*N;
    int vector_count = (total+3) / 4;
    constexpr int threads = 256;
    constexpr int blocks = 256;
    matrix_add_GridStridLoop_float4<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(A),
        reinterpret_cast<const float4*>(B),
        reinterpret_cast<float4*>(C),
        vector_count
    );


    
    cudaDeviceSynchronize();
}
