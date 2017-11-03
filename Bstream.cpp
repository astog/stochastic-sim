#include "bstream.hpp"
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

    uint8_t byte = bytes[byte_index];
    uint8_t bit = (byte >> bit_loc) & 1;
    return bit;   // Shift and mask to find the correct value
}

void stoch::Bstream::set_bit(std::size_t index) {
    // Find the correct byte index, and location within a byte
    std::size_t byte_index = index / 8;
    uint8_t bit_loc = index % 8;

    bytes[byte_index] |= (1<<bit_loc);
}

void stoch::Bstream::reset_bit(std::size_t index) {
    // Find the correct byte index, and location within a byte
    std::size_t byte_index = index / 8;
    uint8_t bit_loc = index % 8;

    bytes[byte_index] &= ~(1<<bit_loc);
}

void stoch::Bstream::toggle_bit(std::size_t index) {
    // Find the correct byte index, and location within a byte
    std::size_t byte_index = index / 8;
    uint8_t bit_loc = index % 8;

    bytes[byte_index] ^= (1<<bit_loc);
}

namespace stoch {
    std::ostream& operator<<(std::ostream& os, const stoch::Bstream& obj) {
        // Go in reverse order so that MSB is on the left, LSB on the right
        std::size_t num_bits = obj.get_length();
        for (std::size_t bit_loc = 0; bit_loc < num_bits; ++bit_loc)
        {
            // Cast to int to print 0,1 not usigned char
            os << (int)obj.get_bit(num_bits-bit_loc-1);
        }

        return os;
    }
}
