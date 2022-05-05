// #include "../common/book.h"
#include "../common/cpu_bitmap.h"
#define DIM 800

// struct for complex numbers
struct cuComplex
{
    float r;
    float i;
    cuComplex(float a, float b) : r(a), i(b) {}

    float magnitude2(void)
    {
        return r * r + i * i;
    }

    cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex z(jx, jy);

    int i = 0;
    // for each point we perform 200 iterations and check whether it diverges or converges
    for (i = 0; i < 200; i++)
    {
        // recursive definition
        z = z * z + c;
        // check if z diverges
        if (z.magnitude2() > 1000)
            return 0;
    }
    // if z converges
    return 1;
}

void kernel(unsigned char *ptr)
{
    // iterates through all point of the 2D matrix and checks if it's in julia set.
    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 4 + 0] = 255 * juliaValue;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }
}

int main(void)
{
    // create a 2D matrix for the Julia Set
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();

    // pass a pointer to the bitmap
    kernel(ptr);

    bitmap.display_and_exit();
}