#include "stochn.hpp"
#include "lfsr.hpp"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

stoch::Stochn::Stochn(uint8_t num, bool randomize, bool rectify) {
    uint8_t seed;
    if (randomize) {
        seed = rand() % bstream_length;
    } else {
        seed = 1;
    }

    polar = false;
    init_bstream(num, seed, rectify);
}

stoch::Stochn::Stochn(int8_t num, bool randomize, bool rectify) {
    uint8_t seed;
    if (randomize) {
        seed = rand() % bstream_length;
    } else {
        seed = 1;
    }

    polar = true;

    // For polar representation 0 => 128 1's and 128 0's
    // All 1's => 128
    // All 0's => -127
    // So offset the abs of the number by +128
    uint8_t polar_num = num >= 0 ? num : -num;
    polar_num += 128;

    init_bstream(polar_num, seed, rectify);
}

stoch::Stochn::Stochn(const Stochn& snum) {
    if (bstream != NULL)
        delete bstream;

    // Create a new bstream, and copy to it
    bstream = new Bstream();
    *bstream = *snum.bstream;
}

stoch::Stochn::~Stochn() {
    delete bstream;
}

void stoch::Stochn::init_bstream(uint8_t num, uint8_t seed, bool rectify) {
    if (bstream != NULL) {
        bstream = new Bstream(bstream_length);
    } else {
        delete bstream;
        bstream = new Bstream(bstream_length);
    }

    // Start with the complemented value, ie consider rectified if we are not rectifying
    bool has_rectified = !rectify;

    stoch::Lfsr lfsr = stoch::Lfsr(seed);
    std::size_t stream_length = bstream -> get_length();

    // Goto each bit and set it based on the seed, and number
    for(std::size_t bit_loc = 0; bit_loc < stream_length; ++bit_loc) {
        if (lfsr.rand() <= num) {
            if (!has_rectified && seed <= num)
                has_rectified = true;   // You only have to rectify one bit
            else
                bstream -> set_bit(bit_loc);
        }
    }
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
