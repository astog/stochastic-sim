#include "bstream.hpp"
#include <iostream>
#include <stdint.h>

namespace stoch {
    class Stochn {
    private:
        Bstream* bstream;

        // Private to prevent direct access to bstream
        Bstream& get_bstream() const {return *bstream;};
    public:
        /* Constructor and Deconstructors */
        Stochn(uint8_t num, bool randomize=true);
        ~Stochn();

        /* Overloaded Operators */
        friend std::ostream& operator<<(std::ostream& os, const Stochn& obj);
    };
}
