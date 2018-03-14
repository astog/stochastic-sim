#ifndef LFSR_HPP
#define LFSR_HPP

#include <stdint.h>

namespace stoch {
    class Lfsr {
    private:
        uint8_t reg;
    public:
        Lfsr();
        Lfsr(uint8_t seed);

        void srand(uint8_t seed);
        uint8_t rand();
    };
}

#endif
