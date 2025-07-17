#ifndef VERSION_H
#define VERSION_H

// Version information for the Factorizer project
#define FACTORIZER_VERSION_MAJOR 2
#define FACTORIZER_VERSION_MINOR 0
#define FACTORIZER_VERSION_PATCH 0

#define FACTORIZER_VERSION_STRING "2.0.0"
#define FACTORIZER_VERSION_DATE "2025-01-17"
#define FACTORIZER_VERSION_CODENAME "Hive-Mind"

// Version check macro
#define FACTORIZER_VERSION_CHECK(major, minor, patch) \
    ((FACTORIZER_VERSION_MAJOR > (major)) || \
     (FACTORIZER_VERSION_MAJOR == (major) && FACTORIZER_VERSION_MINOR > (minor)) || \
     (FACTORIZER_VERSION_MAJOR == (major) && FACTORIZER_VERSION_MINOR == (minor) && \
      FACTORIZER_VERSION_PATCH >= (patch)))

#endif // VERSION_H