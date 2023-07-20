# vector-intrinsics-simd
compute fractal set using vector intrinsic functions (AVX)


## Algorithm
A simple algorithm to generate a representation of the Burningship fractal set is called the “escape time algorithm”. For each pixel in the rendered image, we test using the recurrence relation if the complex number is bounded or not under a maximum number of iterations. Pixels that do not belong to the Mandelbrot set will escape quickly whereas we assume that the pixel is in the set after a fixed maximum number of iterations. A high value of iterations will produce a more detailed image but the computation time will increase accordingly. We use the number of iterations needed to “escape” to depict the pixel value in the image.

When we observe the output image, we are really observing the number of iterations that each pixel took to satisfy a certain constraint. The number of iterations taken is converted to a grayscale value.

## Installation
For implementing this in SIMD I used a machine which happened to supported AVX rather than SSE. The machine I picked from the farm supported AVX256 bit registers so thats what I wrote with the vector functions.

Before compiling, add a directory `mkdir obj` to the simd directory, where the Makefile is located.

### References
https://vovkos.github.io/doxyrest-showcase/opencv/sphinxdoc/page_tutorial_how_to_use_OpenCV_parallel_for_.html

https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/core/how_to_use_OpenCV_parallel_for_/how_to_use_OpenCV_parallel_for_.cpp

https://github.com/OneLoneCoder/Javidx9/blob/master/PixelGameEngine/SmallerProjects/OneLoneCoder_PGE_Mandelbrot.cpp
