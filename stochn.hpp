#include "bstream.hpp"
#include <iostream>
#include <stdint.h>

namespace stoch {
    class Stochn {
    private:
        Bstream* bstream;
        bool polar;  // Does this stream represent a -1/+1 value, instead of a 0/1?

        // Private to prevent direct access to bstream
        Bstream* get_bstream() const {return bstream;};

        void init_bstream(uint8_t num, uint8_t seed, bool rectify);
    public:
        /* Constructors */
        Stochn() : bstream(NULL), polar(false) {};
        Stochn(uint8_t num, bool randomize=true, bool rectify=true);
        Stochn(int8_t num, bool randomize=true, bool rectify=true);
        /* Copy constructor */
        Stochn(const Stochn&);
        /* Deconstructor */
        ~Stochn();

        /* Static global Constants */
        const static std::size_t bstream_length = 256;

        /* Overloaded stream write operator */
        friend std::ostream& operator<<(std::ostream& os, const Stochn& obj);

        Stochn operator*(const Stochn& other);
        Stochn operator+(const Stochn& other);

        /* Accessors and Mutators */
        bool is_polar() const {return polar;};
    };
}
