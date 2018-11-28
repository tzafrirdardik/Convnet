#ifndef _INCLUDER_LIB_
#define _INCLUDER_LIB_
//======================================
//put here all the library that you need.
//======================================

//          microsoft libs
//======================================
#include <ppl.h>
#include <iostream>
#include <time.h>
#include <random>
#include <fstream>
#include <chrono>
#include <amp_math.h>
#include "amp.h"
#include <string>
#include <exception>
#include <vector>
#include <algorithm>
#include <math.h>
//======================================

//      Defined new types and limits
//======================================

//Which data to load, Mnist pictures or Custom pictures.
typedef enum { Custom_, Mnist_ } WhichData_;

//Defined parameters for Matrix.
typedef enum { WithoutZP, WithZP } Padding_;
typedef enum { IsAMatrix, IsAVector } Identity_;
typedef enum { GPU_, CPU_, CPU_parallel_ } Device_;
typedef enum { Zeroes, Random_Val, Ones, DontCare} ElementValues;

//Defined size of filter matrices.
#define Filter_size_R 5
#define Filter_size_C 5

//Defined limits for better performance (CPU, CPU PARALLEL, GPU).
#define FirstLimit 16384    // 128 * 128.
#define SecondLimit 1048576  // 1048 * 1048 * 1.
#define ThirdLimit 2097152 // 128 * 128 * 128. 
#define FourthLimit 16777216 // 4096 * 4096.

//Defined Convnet parameters.
#define LearningRate 0.1f

//Usefull MACROS.
#define STR(x) #x
#define PRINT(x) std::cout << x << std::endl;
#define PRINT_EMPTY_LINE std::cout << std::endl;
//======================================

#endif // !_INCLUDER_LIB_

