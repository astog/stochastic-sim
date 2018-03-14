#include "bstream.hpp"
#include <iostream>
#include <stdint.h>

namespace stoch {
    class Stochn {
    private:
        Bstream* bstream;
        bool polar;  // Does this stream represent a -1/+1 value, instead of a 0/1?
        std::size_t bstream_length;

        // Private to prevent direct access to bstream
        Bstream* get_bstream() const {return bstream;};

        void init_bstream(uint8_t num, uint8_t seed, bool rectify);
    public:
        /* Constructors */
        Stochn() : bstream(NULL), polar(false) {};
        Stochn(uint8_t num, std::size_t length=256, bool polar=false, bool randomize=true, bool rectify=true);
        /* Copy constructor */
        Stochn(const Stochn&);
        /* Deconstructor */
        ~Stochn();

        /* Overloaded stream write operator */
        friend std::ostream& operator<<(std::ostream& os, const Stochn& obj);

        Stochn operator*(const Stochn& other);
        Stochn operator+(const Stochn& other);
        Stochn operator-(const Stochn& other);
        Stochn operator>>(const int val);
        Stochn& operator=(const Stochn& other);

        float to_float() const;
        uint8_t to_count() const { return bstream -> get_bits_set_count();}

        /* Accessors and Mutators */
        bool is_polar() const {return polar;};
    };
}
