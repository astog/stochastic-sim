#include "stochn.hpp"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

stoch::Stochn::Stochn(uint8_t num, bool randomize) {
    std::size_t length = 256;
    uint8_t seed;
    if (randomize) {
        seed = rand() % length;
    } else {
        seed = 1;
    }

    polar = false;
    bstream = new Bstream(length, num, seed);
}

stoch::Stochn::Stochn(int8_t num, bool randomize) {
    std::size_t length = 256;
    uint8_t seed;
    if (randomize) {
        seed = rand() % length;
    } else {
        seed = 1;
    }

    polar = true;
    // For polar representation 0 => 128 1's and 128 0's
    // All 1's => 128
    // All 0's => -127
    // So offset the abs of the number by +128
    uint8_t abs_num = num >= 0 ? num : -num;
    bstream = new Bstream(length, abs_num+128, seed);
}

stoch::Stochn::~Stochn() {
    delete bstream;
}

namespace stoch {
    std::ostream& operator<<(std::ostream& os, const Stochn& obj) {
        int bits_set = obj.get_bstream().get_bits_set_count();
        if (obj.is_polar()) {
            // For polar, do the reverse of the offset during initialization
            return os << 128 - bits_set;
        } else {
            // Since not polar, number of bits is the same as the original number
            return os << bits_set;
        }
    }
}
