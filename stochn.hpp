#include "bstream.hpp"
#include <iostream>
#include <stdint.h>

namespace stoch {
    class Stochn {
    private:
        Bstream* bstream;
        bool polar;  // Does this stream represent a -1/+1 value, instead of a 0/1?

        // Private to prevent direct access to bstream
        Bstream& get_bstream() const {return *bstream;};
    public:
        /* Constructor and Deconstructors */
        Stochn(uint8_t num, bool randomize=true, bool rectify=true);
        Stochn(int8_t num, bool randomize=true, bool rectify=true);
        ~Stochn();

        /* Overloaded Operators */
        friend std::ostream& operator<<(std::ostream& os, const Stochn& obj);

        bool is_polar() const {return polar;};
    };
}
