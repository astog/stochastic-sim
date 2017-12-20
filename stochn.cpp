#include "stochn.hpp"
#include "lfsr.hpp"
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define MUL_METHOD 1
#define ADD_METHOD 2

#define CARRY_MAX_COUNT 3

stoch::Stochn::Stochn(uint8_t num, bool randomize, bool rectify) {
    uint8_t seed;
    if (randomize) {
        seed = rand() % bstream_length;
    } else {
        seed = 28;
    }

    polar = false;
    init_bstream(num, seed, rectify);
}

stoch::Stochn::Stochn(int8_t num, bool randomize, bool rectify) {
    uint8_t seed;
    if (randomize) {
        seed = rand() % bstream_length;
    } else {
        seed = 1;
    }

    polar = true;

    // For polar representation 0 => 128 1's and 128 0's
    // All 1's => 128
    // All 0's => -127
    // So offset the abs of the number by +128
    uint8_t polar_num = num >= 0 ? num : -num;
    polar_num += 128;

    init_bstream(polar_num, seed, rectify);
}

stoch::Stochn::Stochn(const Stochn& snum) {
    // std::cout << "Copying " << &snum << " into " << this << "\n";

    // if (bstream != NULL)
    //     delete bstream;

    // Create a new bstream, and clone to it
    polar = snum.polar;
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
        return stoch::Stochn((uint8_t)0); // return 0 stream
    }

    if (lhs_polar) {
        stoch::Stochn result = stoch::Stochn((int8_t)0);

        // TODO: Add multiplication for polar

        return result;
    } else {

        stoch::Stochn result = stoch::Stochn((uint8_t)0);
        Bstream* result_bstream = result.get_bstream();   // Bstream is passed by reference, so changes go back to original
        Bstream* lhs_bstream = this -> get_bstream();
        Bstream* rhs_bstream = other.get_bstream();

        // std::cout << *lhs_bstream << std::endl << std::endl;
        // std::cout << *rhs_bstream << std::endl << std::endl;

        #if (MUL_METHOD == 1)
            // Multiplication is basic ANDing LHS with RHS
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
        return stoch::Stochn((uint8_t)0); // return 0 stream
    }

    if (lhs_polar) {
        stoch::Stochn result = stoch::Stochn((int8_t)0);

        // TODO: Add multiplication for polar

        return result;
    } else {

        stoch::Stochn result = stoch::Stochn((uint8_t)0);
        #if (ADD_METHOD == 1)
            stoch::Stochn mux_select = stoch::Stochn((uint8_t)128);

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
        return stoch::Stochn((uint8_t)0); // return 0 stream
    }

    if (lhs_polar) {
        stoch::Stochn result = stoch::Stochn((int8_t)0);

        // TODO: Add multiplication for polar

        return result;
    } else {

        stoch::Stochn result = stoch::Stochn((uint8_t)0);

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
    stoch::Stochn sn1 = stoch::Stochn(uint8_t((1.0/(1<<val))*255));
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
        int8_t val = 128 - bits_set;
        if (val < 0)
            return val / 127.0;
        else
            return val / 128.0;
    } else {
        return bits_set / 255.0;
    }
}

// Operators
namespace stoch {
    std::ostream& operator<<(std::ostream& os, const Stochn& obj) {
        int bits_set = obj.to_count();
        if (obj.is_polar()) {
            // For polar, do the reverse of the offset during initialization
            return os << 128 - bits_set;
        } else {
            // Since not polar, number of bits is the same as the original number
            return os << bits_set;
        }
    }
}
