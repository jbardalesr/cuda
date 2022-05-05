#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
    float r;
    float i;

    __device__ __host__
    cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2(void)
    {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

// __device__ functions can be called only from the device, and it is execute only in the device
__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex z(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        z = z * z + c;
        if (z.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

// __global__ functions can be called from the host or device, and it is execute only in the device
__global__ void kernel(unsigned char *ptr)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main(void)
{
    // observe that bitmap and grid of blocks have the same dimension
    // create an empty 2D matrix for the Julia Set
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    // allocate the image memory on the GPU
    HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

    // specify a two dimensional blocks
    dim3 grid(DIM, DIM);

    // pass our dim3 variable grid to the CUDA runtime
    kernel<<<grid, 1>>>(dev_bitmap);

    // copy the results on the CPU
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),
                            dev_bitmap,
                            bitmap.image_size(),
                            cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
}