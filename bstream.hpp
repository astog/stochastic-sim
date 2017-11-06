#ifndef BSTREAM_HPP
#define BSTREAM_HPP

#include <stdint.h>
#include <cstddef>
#include <iostream>

namespace stoch {
    class Bstream {
    private:
        uint8_t* bytes;
        std::size_t stream_length;

    public:
        /* Constructor and Deconstructors */
        Bstream(std::size_t length);
        Bstream(std::size_t length, uint8_t seed, uint8_t num, bool rectify=true);
        ~Bstream();

        /* Mutators */
        uint8_t get_bit(std::size_t index) const;
        std::size_t get_accum_count() const;
        void set_bit(std::size_t index);
        void reset_bit(std::size_t index);
        void toggle_bit(std::size_t index);
        std::size_t get_length() const {return stream_length;};

        /* Overloaded Operators */
        friend std::ostream& operator<<(std::ostream& os, const Bstream& obj);
    };
}

#endif
