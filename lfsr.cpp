#include "lfsr.hpp"

#define TAP1_LOC 0
#define TAP2_LOC 2
#define TAP3_LOC 3
#define TAP4_LOC 5

#define SHIFT_IN_LOC 7

stoch::Lfsr::Lfsr() {
    reg = 1;
}

stoch::Lfsr::Lfsr(uint8_t seed) {
    reg = seed;
}

void stoch::Lfsr::srand(uint8_t seed) {
    reg = seed;
}

uint8_t stoch::Lfsr::rand() {
    uint8_t return_reg = reg;
    // Taps at 0th and 1st bit
    uint8_t shift_in_bit = 0;

    #ifdef TAP1_LOC
        shift_in_bit ^= ((reg >> TAP1_LOC) & 1);
    #endif

    #ifdef TAP2_LOC
        shift_in_bit ^= ((reg >> TAP2_LOC) & 1);
    #endif

    #ifdef TAP3_LOC
        shift_in_bit ^= ((reg >> TAP3_LOC) & 1);
    #endif

    #ifdef TAP4_LOC
        shift_in_bit ^= ((reg >> TAP4_LOC) & 1);
    #endif

    reg = (reg >> 1) | (shift_in_bit << SHIFT_IN_LOC);
    return return_reg;
}
