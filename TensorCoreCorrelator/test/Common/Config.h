#if !defined CONFIG_H
#define CONFIG_H

#if defined __ARM_ARCH
#define UNIFIED_MEMORY // assume this is a Jetson Xavier
#endif

#undef MEASURE_POWER

#define POL_X			0
#define POL_Y			1
#define NR_POLARIZATIONS	2

#if !defined NR_TIMES
#define NR_TIMES		768
#endif

#define REAL			0
#define IMAG			1
#define COMPLEX			2

#endif
