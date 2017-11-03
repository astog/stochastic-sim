#include "bstream.hpp"
#include "bit_ops.h"
#include <math.h>

stoch::Bstream::Bstream(std::size_t length) {
    std::size_t num_bytes = ceil(length/8.0);
    bytes = new uint8_t[num_bytes];
    stream_length = length;
}

stoch::Bstream::~Bstream() {
    delete[] bytes;
}

uint8_t stoch::Bstream::get_bit(std::size_t index) const {
    // Find the correct byte index, and location within a byte
    std::size_t byte_index = index / 8;
    uint8_t bit_loc = index % 8;

    return GET_BIT(bytes[byte_index], bit_loc);
}

void stoch::Bstream::set_bit(std::size_t index) {
    // Find the correct byte index, and location within a byte
    std::size_t byte_index = index / 8;
    uint8_t bit_loc = index % 8;

    SET_BIT(bytes[byte_index], bit_loc);
}

void stoch::Bstream::reset_bit(std::size_t index) {
    // Find the correct byte index, and location within a byte
    std::size_t byte_index = index / 8;
    uint8_t bit_loc = index % 8;

    RESET_BIT(bytes[byte_index], bit_loc);
}

void stoch::Bstream::toggle_bit(std::size_t index) {
    // Find the correct byte index, and location within a byte
    std::size_t byte_index = index / 8;
    uint8_t bit_loc = index % 8;

    TOGGLE_BIT(bytes[byte_index], bit_loc);
}

namespace stoch {
    std::ostream& operator<<(std::ostream& os, const stoch::Bstream& obj) {
        // Go in reverse order so that MSB is on the left, LSB on the right
        std::size_t num_bits = obj.get_length();
        for (std::size_t bit_loc = 0; bit_loc < num_bits; ++bit_loc)
        {
            // Cast to int to print 0,1 not unsigned char
            os << (int)obj.get_bit(num_bits-bit_loc-1);
        }

        return os;
    }
}
