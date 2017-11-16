#include "stochn.hpp"
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>

float float_to_fixedp(float n) {
    return (round(n*255))/255.0;
}

int main() {
    srand(time(0));

    // Start with random value (155-255)
    uint8_t n = 150;
    float val = n/255.0;
    stoch::Stochn sn = stoch::Stochn(n, true);
    std::cout << sn << ", " << (int)(val*255) << std::endl;

    for (uint8_t n_new = 200; n_new < 220; n_new++) {
        stoch::Stochn sn_new = stoch::Stochn(n_new, true);
        // std::cout << sn << " * " << sn_new << std::endl;
        sn = sn * sn_new;
        val = float_to_fixedp(val * n_new/255.0);
        std::cout << sn << ", " << (int)(val*255) << std::endl;
    }

    /*
    stoch::Stochn sn1 = stoch::Stochn((uint8_t)55);
    stoch::Stochn sn2 = stoch::Stochn((uint8_t)67);
    std::cout << sn1 + sn2 << std::endl;
    */

    return 0;
}
