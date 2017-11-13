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
        /* Constructors */
        Bstream() : bytes(NULL), stream_length(0) {};
        Bstream(std::size_t length);
        /* Copy Constructor */
        Bstream(const Bstream&);
        /* Deconstructor */
        ~Bstream();

        /* Accessors and Mutators */
        uint8_t get_bit(std::size_t index) const;
        std::size_t get_bits_set_count() const;
        void set_bit(std::size_t index);
        void reset_bit(std::size_t index);
        void toggle_bit(std::size_t index);
        std::size_t get_length() const {return stream_length;};

        /* Write to stream overloaded operator */
        friend std::ostream& operator<<(std::ostream& os, const Bstream& obj);
    };
}

#endif
