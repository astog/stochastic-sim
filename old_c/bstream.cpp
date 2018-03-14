#include "bstream.hpp"
#include "bit_ops.h"
#include "lfsr.hpp"
#include <math.h>
// #include <stdio.h>

stoch::Bstream::Bstream(std::size_t length) {
    std::size_t num_bytes = ceil(length/8.0);
    bytes = new uint8_t[num_bytes];
    stream_length = length;

    // Zero each byte
    for (std::size_t byte_loc = 0; byte_loc < num_bytes; byte_loc++) {
        bytes[byte_loc] = 0;
    }
}

stoch::Bstream::Bstream(const Bstream& bstream) {
    // Delete any previous bytes
    if (bytes != NULL)
        delete[] bytes;

    // Create stream with same length
    std::size_t num_bytes = ceil(bstream.stream_length/8.0);
    bytes = new uint8_t[num_bytes];
    stream_length = bstream.stream_length;

    // Copy each byte
    for (std::size_t byte_loc = 0; byte_loc < num_bytes; byte_loc++) {
        bytes[byte_loc] = bstream.bytes[byte_loc];
    }
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

std::size_t stoch::Bstream::get_bits_set_count() const {
    std::size_t count = 0;

    // Create lookup table (static since this is constant throughout operation)
    // Source: https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
    static const unsigned char BitsSetTable256[256] = {
    #   define B2(n) n,     n+1,     n+1,     n+2
    #   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
    #   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
        B6(0), B6(1), B6(1), B6(2)
    };

    std::size_t num_bytes = ceil(stream_length/8.0);
    for(std::size_t byte_index = 0; byte_index < num_bytes; ++byte_index) {
        count += BitsSetTable256[bytes[byte_index]];
    }

    return count;
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
    std::ostream& operator<<(std::ostream& os, const Bstream& obj) {
        // Go in reverse order so that MSB is on the left, LSB on the right
        std::size_t num_bits = obj.get_length();
        for (std::size_t bit_loc = 0; bit_loc < num_bits; ++bit_loc) {
            // Cast to int to print 0,1 not unsigned char
            os << (int)obj.get_bit(num_bits-bit_loc-1);
        }

        return os;
    }
}

stoch::Bstream* stoch::Bstream::clone() {
    Bstream* cloned = new Bstream(stream_length);

    // Copy each bit
    std::size_t num_bits = stream_length;
    for (std::size_t bit_loc = 0; bit_loc < num_bits; ++bit_loc) {
        if (get_bit(bit_loc)) {
            cloned -> set_bit(bit_loc);
        }
    }

    return cloned;
}
