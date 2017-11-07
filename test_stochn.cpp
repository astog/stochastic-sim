#include "stochn.hpp"

int main() {
    stoch::Stochn s1 = stoch::Stochn(64);
    stoch::Stochn s2 = stoch::Stochn(128);

    std::cout << s1 << std::endl << s2 << std::endl;

    return 0;
}
