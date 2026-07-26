#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height){
    //像素索引和像素总量
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_count = width * height;
    if(pixel_idx>=pixel_count) return;
    uchar4* pixels = reinterpret_cast<uchar4*>(image);
    uchar4 pixel = pixels[pixel_idx];
    pixel.x = 255 - pixel.x;
    pixel.y = 255 - pixel.y;
    pixel.z = 255 - pixel.z;
    pixels[pixel_idx] = pixel;
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
