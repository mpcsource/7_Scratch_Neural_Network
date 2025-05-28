#include "math/random.hpp"

// # Returns a random number between 0 and 9.
int random10Int() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    unsigned int seed = ts.tv_nsec;
    unsigned int result = (1103515245 * seed + 12345) % 10;
    unsigned int mask = result >> 31;
    return (result + mask);
}

// # Weirdest random number in range generator.
float randomRange(float min, float max) {
    srand(random10Int() + 10 * random10Int() + 100 * random10Int() + 0.1 * random10Int());
    return min + (rand() / (float) RAND_MAX) * (max-min);
}