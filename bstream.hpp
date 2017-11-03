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
        Bstream(std::size_t length);
        ~Bstream();

        uint8_t get_bit(std::size_t index) const;
        void set_bit(std::size_t index);
        void reset_bit(std::size_t index);
        void toggle_bit(std::size_t index);
        std::size_t get_length() const {return stream_length;};

        friend std::ostream& operator<<(std::ostream& os, const Bstream& obj);
    };
}

#endif
