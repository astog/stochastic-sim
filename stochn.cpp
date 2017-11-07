#include "stochn.hpp"
#include <stdlib.h>
#include <time.h>

stoch::Stochn::Stochn(uint8_t num, bool randomize) {
    std::size_t length = 256;
    uint8_t seed;
    if (randomize) {
        seed = rand() % length;
    } else {
        seed = 1;
    }

    bstream = new Bstream(length, num, seed, false);
}

stoch::Stochn::~Stochn() {
    delete bstream;
}

namespace stoch {
    std::ostream& operator<<(std::ostream& os, const Stochn& obj) {
        // The number is just the number of bits set
        return os << obj.get_bstream().get_bits_set_count();
    }
}
