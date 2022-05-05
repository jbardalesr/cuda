#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include "../common/cpu_anim.h"

#define DIM 256

struct DataBlock
{
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

__global__ void kernel(unsigned char *ptr, int ticks)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position centered
    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrt(fx * fx + fy * fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

    // RGBA for each pixel of the image DIMxDIM
    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}

// clean up memory allocated on the GPU
void cleanup(DataBlock *d)
{
    cudaFree(d->dev_bitmap);
}

// this function will be called by the main class every time it wants to generate a new frame of the animatation
void generate_frame(DataBlock *d, int ticks)
{
    // declare two dimensional variables
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    // each thread will have a unique (x, y) of the image
    kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
                            d->dev_bitmap,
                            d->bitmap->image_size(),
                            cudaMemcpyDeviceToHost));

}

int main(void)
{
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR(cudaMalloc((void **)&data.dev_bitmap, bitmap.image_size()));

    bitmap.anim_and_exit((void (*)(void *, int))generate_frame, (void (*)(void *))cleanup);
}