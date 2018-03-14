#include <stdio.h>
#include "bstream.hpp"
#include "lfsr.hpp"
// #include "bit_ops.h"

#define RECTIFY_LFSR_WSEED 1

int main(int argc, char const *argv[]) {
    printf("%s, %s, %s\n", "Number", "Seed", "LFSR count");
    for (int num = 0; num < 256; ++num) {
        for (int seed = 1; seed <= 255; ++seed) {
            stoch::Lfsr l1 = stoch::Lfsr(seed);
            int count = 0;
            for (int i = 0; i < 256; ++i) {
                uint8_t n = l1.rand();
                if (n <= num)
                    count++;
            }

            #if (RECTIFY_LFSR_WSEED == 1)
                // Add one, based on seed
                if (seed <= num)
                    count--;
            #endif

            printf("%d, %d, %d\n", num, seed, count);
        }
    }

    return 0;
}
