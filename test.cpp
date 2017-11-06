#include <iostream>
#include "bstream.hpp"

int main(int argc, char const *argv[])
{
    stoch::Bstream b2 = stoch::Bstream(256, 128, 28);
    std::cout << b2 << std::endl;
    std::cout << b2.get_accum_count() << std::endl;

    stoch::Bstream b3 = stoch::Bstream(256, 128, 9, false);
    std::cout << b3 << std::endl;
    std::cout << b3.get_accum_count() << std::endl;

    return 0;
}
