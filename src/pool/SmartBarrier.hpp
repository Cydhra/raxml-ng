#ifndef RAXML_SMARTBARRIER_HPP_
#define RAXML_SMARTBARRIER_HPP_

#include <pthread.h>

/**
 * A unique smart pointer for pthread_barrier_t that handles initialization and deletion automatically,
 * and therefore upholds a RAII contract and move construction.
 *
 * The smart barrier is more expensive than the standard barrier used by RAxML-ng, but is platform-independent,
 * well-defined, and re-entry-safe.
 */
class SmartBarrier : std::unique_ptr<pthread_barrier_t, std::function<void(pthread_barrier_t *)> > {
public:
    explicit SmartBarrier(const unsigned int num_threads) : unique_ptr(new pthread_barrier_t, [](pthread_barrier_t *b) {
        if (b) {
            pthread_barrier_destroy(b);
            delete b;
        }
    }) {
        pthread_barrier_init(this->get(), nullptr, num_threads);
    }

    using unique_ptr::operator*;

    using unique_ptr::operator->;

    void enter() const noexcept {
        pthread_barrier_wait(this->get());
    }
};

#endif //RAXML_SMARTBARRIER_HPP_
