#include "stochn.hpp"
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(0));

    stoch::Stochn s1 = stoch::Stochn((uint8_t)64);
    stoch::Stochn s2 = stoch::Stochn((uint8_t)128);

    std::cout << s1 << std::endl << s2 << std::endl;

    stoch::Stochn s3 = stoch::Stochn((int8_t)-127); // Polar stochastic number
    std::cout << s3 << std::endl;

    std::cout << (s1 * s2) << std::endl;

    return 0;
}
