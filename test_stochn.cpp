#include "stochn.hpp"
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(0));

    stoch::Stochn s1 = stoch::Stochn(64);
    stoch::Stochn s2 = stoch::Stochn(128);

    std::cout << s1 << std::endl << s2 << std::endl;

    return 0;
}
