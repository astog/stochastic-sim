#include "stochn.hpp"
#include "lfsr.hpp"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <stdio.h>

#define MUL_METHOD 1
#define ADD_METHOD 2

#define CARRY_MAX_COUNT 3

stoch::Stochn::Stochn(uint8_t num, std::size_t length, bool polar, bool randomize, bool rectify) {
    uint8_t seed;
    bstream_length = length;
    if (randomize) {
        seed = rand() % 256;
    } else {
        seed = 28;
    }

    this->polar = polar;
    init_bstream(num, seed, rectify);
}

stoch::Stochn::Stochn(const Stochn& snum) {
    // std::cout << "Copying " << &snum << " into " << this << "\n";

    // if (bstream != NULL)
    //     delete bstream;

    // Create a new bstream, and clone to it
    polar = snum.polar;
    bstream_length = snum.bstream_length;
    bstream = snum.bstream -> clone();
}

stoch::Stochn::~Stochn() {
    // std::cout << "Calling deconstructor on  " << this << "\n";
    if (bstream != NULL)
        delete bstream;
}

void stoch::Stochn::init_bstream(uint8_t num, uint8_t seed, bool rectify) {
    if (bstream != NULL) {
        bstream = new Bstream(bstream_length);
    } else {
        delete bstream;
        bstream = new Bstream(bstream_length);
    }

    // Start with the complemented value, ie consider rectified if we are not rectifying
    bool has_rectified = !rectify;

    stoch::Lfsr lfsr = stoch::Lfsr(seed);
    std::size_t stream_length = bstream -> get_length();

    // Goto each bit and set it based on the seed, and number
    for(std::size_t bit_loc = 0; bit_loc < stream_length; ++bit_loc) {
        if (lfsr.rand() <= num) {
            if (!has_rectified && seed <= num)
                has_rectified = true;   // You only have to rectify one bit
            else
                bstream -> set_bit(bit_loc);
        }
    }
}

stoch::Stochn stoch::Stochn::operator*(const stoch::Stochn& other) {
    // std::cout << "Calling operator: " << this << " * " << &other << "\n";
    // Checks for polarity, both should be the same
    bool lhs_polar = this -> is_polar();
    bool rhs_polar = other.is_polar();
    if (lhs_polar != rhs_polar) {
        return stoch::Stochn((uint8_t)0, bstream_length); // return 0 stream
    }

    if (lhs_polar) {
        stoch::Stochn result = stoch::Stochn(0, bstream_length, true);

        Bstream* result_bstream = result.get_bstream();   // Bstream is passed by reference, so changes go back to original
        Bstream* lhs_bstream = this -> get_bstream();
        Bstream* rhs_bstream = other.get_bstream();

        // std::cout << *lhs_bstream << std::endl << std::endl;
        // std::cout << *rhs_bstream << std::endl << std::endl;

        #if (MUL_METHOD == 1)
            // Multiplication is basic LHS XNOR RHS
            std::size_t stream_length = result_bstream -> get_length();
            for(std::size_t bit_loc = 0; bit_loc < stream_length; ++bit_loc) {
                if (lhs_bstream->get_bit(bit_loc) == 0 & rhs_bstream->get_bit(bit_loc) == 0) {
                    //printf("(%lu): %d %d\n", bit_loc, lhs_bstream->get_bit(bit_loc), rhs_bstream->get_bit(bit_loc));
                    result_bstream -> set_bit(bit_loc);
                } else if (lhs_bstream->get_bit(bit_loc) == 1 & rhs_bstream->get_bit(bit_loc) == 1) {
                    // printf("(%lu): %d %d\n", bit_loc, lhs_bstream->get_bit(bit_loc), rhs_bstream->get_bit(bit_loc));
                    result_bstream -> set_bit(bit_loc);
                }
            }
        #endif

        // std::cout << *result_bstream << std::endl << std::endl;
        return result;
    } else {

        stoch::Stochn result = stoch::Stochn((uint8_t)0, bstream_length);
        Bstream* result_bstream = result.get_bstream();   // Bstream is passed by reference, so changes go back to original
        Bstream* lhs_bstream = this -> get_bstream();
        Bstream* rhs_bstream = other.get_bstream();

        // std::cout << *lhs_bstream << std::endl << std::endl;
        // std::cout << *rhs_bstream << std::endl << std::endl;

        #if (MUL_METHOD == 1)
            // Multiplication is basic LHS AND RHS
            std::size_t stream_length = result_bstream -> get_length();
            for(std::size_t bit_loc = 0; bit_loc < stream_length; ++bit_loc) {
                if (lhs_bstream -> get_bit(bit_loc) & rhs_bstream -> get_bit(bit_loc)) {
                    result_bstream -> set_bit(bit_loc);
                }
            }
        #endif
        // std::cout << "\n\n";

        // std::cout << *result_bstream << std::endl << std::endl;
        // std::cout << "Returning " << &result << std::endl;
        return result;
    }
}

stoch::Stochn stoch::Stochn::operator+(const stoch::Stochn& other) {
    // std::cout << "Calling operator: " << this << " * " << &other << "\n";
    // Checks for polarity, both should be the same
    bool lhs_polar = this -> is_polar();
    bool rhs_polar = other.is_polar();
    if (lhs_polar != rhs_polar) {
        return stoch::Stochn((uint8_t)0, bstream_length); // return 0 stream
    }

    if (lhs_polar) {
        stoch::Stochn result = stoch::Stochn((uint8_t)0, bstream_length, true);

    } else {

        stoch::Stochn result = stoch::Stochn((uint8_t)0, bstream_length);
        #if (ADD_METHOD == 1)
            stoch::Stochn mux_select = stoch::Stochn((uint8_t)127, bstream_length, true); // 0 in polar format

            // Extract bstreams
            Bstream* result_bstream = result.get_bstream();   // Bstream is passed by reference, so changes go back to original
            Bstream* mux_select_bstream = mux_select.get_bstream();
            Bstream* lhs_bstream = this -> get_bstream();
            Bstream* rhs_bstream = other.get_bstream();

            // std::cout << *lhs_bstream << std::endl;
            // std::cout << *rhs_bstream << std::endl;
            // std::cout << *mux_select_bstream << std::endl;

            // Addition is muxing LHS RHS
            std::size_t stream_length = result_bstream -> get_length();
            for(std::size_t bit_loc = 0; bit_loc < stream_length; ++bit_loc) {
                if (mux_select_bstream -> get_bit(bit_loc)) {
                    if (rhs_bstream -> get_bit(bit_loc))
                        result_bstream -> set_bit(bit_loc);
                } else {
                    if (lhs_bstream -> get_bit(bit_loc))
                        result_bstream -> set_bit(bit_loc);
                }
            }

            // std::cout << *result_bstream << std::endl;
        #elif (ADD_METHOD == 2)
            // Extract bstreams
            Bstream* result_bstream = result.get_bstream();   // Bstream is passed by reference, so changes go back to original
            Bstream* lhs_bstream = this -> get_bstream();
            Bstream* rhs_bstream = other.get_bstream();

            // Addition is keeping memory of AND, adding 1's to OR stream
            uint8_t carry_count = 0;
            std::size_t stream_length = result_bstream -> get_length();
            for(std::size_t bit_loc = 0; bit_loc < stream_length; ++bit_loc) {
                if (lhs_bstream -> get_bit(bit_loc) | rhs_bstream -> get_bit(bit_loc)) {
                    if (lhs_bstream -> get_bit(bit_loc) & rhs_bstream -> get_bit(bit_loc)) {
                        if (carry_count < CARRY_MAX_COUNT) {
                            carry_count ++;
                        }
                    }
                    result_bstream -> set_bit(bit_loc);
                } else {
                    if (carry_count > 0) {
                        carry_count--;
                        result_bstream -> set_bit(bit_loc);
                    }
                }
            }
        #endif

        // std::cout << "Returning " << &result << std::endl;
        return result;
    }
}

stoch::Stochn stoch::Stochn::operator-(const stoch::Stochn& other) {
    // std::cout << "Calling operator: " << this << " * " << &other << "\n";
    // Checks for polarity, both should be the same
    bool lhs_polar = this -> is_polar();
    bool rhs_polar = other.is_polar();
    if (lhs_polar != rhs_polar) {
        return stoch::Stochn((uint8_t)0, bstream_length); // return 0 stream
    }

    if (lhs_polar) {
        stoch::Stochn result = stoch::Stochn((int8_t)0, bstream_length, true);

        // TODO: Add multiplication for polar

        return result;
    } else {

        stoch::Stochn result = stoch::Stochn((uint8_t)0, bstream_length);

        // Extract bstreams
        Bstream* result_bstream = result.get_bstream();   // Bstream is passed by reference, so changes go back to original
        Bstream* lhs_bstream = this -> get_bstream();
        Bstream* rhs_bstream = other.get_bstream();

        // Subtraction is every 1 in rhs should not be in lhs
        int carry_count = 0;
        std::size_t stream_length = result_bstream -> get_length();
        for(std::size_t bit_loc = 0; bit_loc < stream_length; ++bit_loc) {
            // We set if rhs is 0 and lhs is 1
            if (lhs_bstream->get_bit(bit_loc) & ~rhs_bstream->get_bit(bit_loc)) {
                if (carry_count == 0)
                    result_bstream -> set_bit(bit_loc);
                else
                    carry_count--;
            } else if (~lhs_bstream->get_bit(bit_loc) & rhs_bstream->get_bit(bit_loc)) {
                if (carry_count < CARRY_MAX_COUNT)
                    carry_count++;
            }
        }

        // std::cout << *result_bstream << std::endl;

        // std::cout << "Returning " << &result << std::endl;
        return result;
    }
}

stoch::Stochn stoch::Stochn::operator>>(const int val) {
    // Find the correct multipler
    stoch::Stochn sn1 = stoch::Stochn(uint8_t((1.0/(1<<val))*255), bstream_length);
    // std::cout << "Muliplying with " << sn1.to_float() << std::endl;
    return (*this)*sn1;
}

stoch::Stochn& stoch::Stochn::operator=(const stoch::Stochn& other) {
    // std::cout << "Calling operator: " << this << " = " << &other << "\n";
    if (bstream != NULL)
        delete bstream;

    // Create a new bstream, and clone to it
    bstream = other.bstream -> clone();
    // std::cout << "Returning " << this << std::endl;
    return *this;
}

float stoch::Stochn::to_float() const {
    int bits_set = bstream -> get_bits_set_count();
    if (polar) {
        int bits_nset = bstream_length-bits_set;
        return ((float)(bits_set-bits_nset))/bstream_length;
    } else {
        return bits_set / (bstream_length);
    }
}

// Operators
namespace stoch {
    std::ostream& operator<<(std::ostream& os, const Stochn& obj) {
        int bits_set = obj.to_count();
        if (obj.is_polar()) {
            // For polar, do the reverse of the offset during initialization
            return os << bits_set - 127;
        } else {
            // Since not polar, number of bits is the same as the original number
            return os << bits_set;
        }
    }
}
