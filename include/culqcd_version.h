#define CULQCD_VERSION_MAJOR 0
#define CULQCD_VERSION_MINOR 1
#define CULQCD_VERSION_SUBMINOR 0

/**
 * @def   CULQCD_VERSION
 * @brief This macro is deprecated.  Use CULQCD_VERSION_MAJOR, etc., instead.
 */
#define CULQCD_VERSION                                                         \
  ((CULQCD_VERSION_MAJOR << 16) | (CULQCD_VERSION_MINOR << 8) |                \
   CULQCD_VERSION_SUBMINOR)
