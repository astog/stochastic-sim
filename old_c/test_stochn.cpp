#include "stochn.hpp"
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>

#define TEST 6

float float_to_fixedp(float n) {
    return (round(n*255))/255.0;
}

void print_float_binary(float f) {\
    int bit = 0;
    int *b = reinterpret_cast<int*>(&f);
    for (int k = 31; k >=0; k--)
    {
        bit = ((*b >> k)&1);
        std::cout << bit;
    }
}

int main() {
    srand(time(0));

    #if (TEST==1)

    // Start with random value (155-255)
    uint8_t n = 150;
    float val = n/255.0;
    stoch::Stochn sn = stoch::Stochn(n);
    std::cout << sn << ", " << (int)(val*255) << std::endl;

    for (uint8_t n_new = 200; n_new < 220; n_new++) {
        stoch::Stochn sn_new = stoch::Stochn(n_new);
        // std::cout << sn << " * " << sn_new << std::endl;
        sn = sn * sn_new;
        val = float_to_fixedp(val * n_new/255.0);
        std::cout << sn << ", " << (int)(val*255) << std::endl;
    }

    #elif (TEST==2)

    for (std::size_t i = 0; i < 100; i++) {
        uint8_t x = rand()%256;
        uint8_t y = rand()%256;
        stoch::Stochn sn1 = stoch::Stochn(x);
        stoch::Stochn sn2 = stoch::Stochn(y);
        std::cout << sn1+sn2 << " " << std::min(x+y,255) << std::endl;
    }

    #elif (TEST==3)

    for (std::size_t i = 0; i < 10; i++) {
        uint8_t x = rand()%256;
        int sh_amt = rand()%9;
        stoch::Stochn sn1 = stoch::Stochn(x);
        std::cout << (sn1>>sh_amt) << " " << ((int)x >> sh_amt) << std::endl;
    }

    #elif (TEST==4)

    for (std::size_t i = 0; i < 100; i++) {
        uint8_t x = rand()%128;
        uint8_t y = 128+(rand()%128);
        stoch::Stochn sn1 = stoch::Stochn(x);
        stoch::Stochn sn2 = stoch::Stochn(y);
        std::cout << (sn2-sn1) << " " << (int)y << "-" << (int)x  << "=" << (y-x) << std::endl;
    }

    #elif (TEST==5)

    for (int i = 1; i <= 10; i++)
    {
        uint8_t x = rand()%128;
        uint8_t y = rand()%128;
        stoch::Stochn sn1 = stoch::Stochn(x, 256, true); // std::cout << sn1.to_float() << std::endl;
        stoch::Stochn sn2 = stoch::Stochn(y, 256, true); // std::cout << sn2.to_float() << std::endl;
        // std::cout << (int)sn1.to_count() << ": " << sn1.to_float() << std::endl;
        // std::cout << (int)sn2.to_count() << ": " << sn2.to_float() << std::endl;
        std::cout << (sn1*sn2).to_float() << " " << sn1.to_float()*sn2.to_float() << std::endl;
    }

    #elif (TEST==6)

    uint8_t vals[256];

    for (int i = 1; i < 256*10; i++)
    {
        uint8_t x = rand%256;
        stoch::Stochn sn1 = stoch::Stochn(x, 256);
        float real_n = sn1.to_float();
        uint8_t approx_x = real_n * 256
    }

    #endif

    return 0;
}
