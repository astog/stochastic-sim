#ifndef BIT_OPS_H
#define BIT_OPS_H

#define SET_BIT(VAL,LOC)    (VAL |= 1<<LOC)
#define RESET_BIT(VAL,LOC)  (VAL &= ~(1<<LOC))
#define TOGGLE_BIT(VAL,LOC) (VAL ^= 1<<LOC)
#define GET_BIT(VAL,LOC)    ((VAL >> LOC) & 1)

#define PRINT_BINARY(N, type){\
    type __temp = N;\
    type ___mask = ((type)1)<<(sizeof(type)*8-1);\
    for(std::size_t ___i=0; ___i < sizeof(type)*8; ++___i) {\
        if(__temp & ___mask)\
            printf("1");\
        else\
            printf("0");\
        __temp <<= ((type)1);\
    }\
}

#endif
