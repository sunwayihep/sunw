/* Temporarily remove CULQCD's `#define complex` (see complex.h) while parsing
 * Thrust / CUB / CCCL headers, which use `complex` as an identifier.
 * Pair every include of this file with culqcd_cccl_guard_end.h after the last
 * such third-party #include in the same contiguous block. */

#ifdef complex
#pragma push_macro("complex")
#undef complex
#define CULQCD_CCCL_GUARD_COMPLEX_ACTIVE
#endif
