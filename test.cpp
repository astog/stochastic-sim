#include <iostream>
#include "bstream.hpp"

int main(int argc, char const *argv[])
{
    stoch::Bstream b1 = stoch::Bstream(65);
    std::cout << b1 << std::endl;

    b1.set_bit(0);
    b1.set_bit(1);
    b1.set_bit(2);
    b1.set_bit(3);
    b1.set_bit(5);

    b1.toggle_bit(0);
    b1.reset_bit(2);

    b1.set_bit(64);

    std::cout << b1 << std::endl;
    return 0;
}
