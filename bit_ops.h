#ifndef BIT_OPS_H
#define BIT_OPS_H

#define SET_BIT(VAL,LOC)    (VAL |= 1<<LOC)
#define RESET_BIT(VAL,LOC)  (VAL &= ~(1<<LOC))
#define TOGGLE_BIT(VAL,LOC) (VAL ^= 1<<LOC)
#define GET_BIT(VAL,LOC)    ((VAL >> LOC) & 1)

#endif
