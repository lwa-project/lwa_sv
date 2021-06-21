#if !defined EXCEPTION_PROPAGATOR_H
#define EXCEPTION_PROPAGATOR_H

// Experimental class to propagate exceptions around OpenMP pragmas (and other
// constructs) that cannot cope with exceptions.  Exceptions thrown inside a lambda
// function that is given to the () operator are propagated and rethrown when the
// progagation object is destroyed.  Example use case:
//
// ExceptionPropagator ep;
//
// #pragma omp parallel for
// for (int i = 0; i < 10000; i ++)
//   if (!ep) // finish loop ASAP if exception is pending
//     ep([&] () {
//       throw 42; // continue with code that might throw exceptions
//     });
//
// // exception is rethrown at scope exit of ep

#include <atomic>
#include <exception>


class ExceptionPropagator
{
  public:
    ExceptionPropagator()
    :
      propagateException(ATOMIC_FLAG_INIT)
    {
    }

    ~ExceptionPropagator() noexcept(false)
    {
      if (exception != nullptr && !std::uncaught_exception())
        std::rethrow_exception(exception);
    }

    template <typename T> void operator () (const T &func)
    {
      try {
        func();
      } catch (...) {
        if (!atomic_flag_test_and_set(&propagateException))
          exception = std::current_exception();
      }
    }

    operator bool () const // returns true iff exception pending
    {
      return exception != nullptr;
    }

  private:
    std::atomic_flag   propagateException;
    std::exception_ptr exception;
};

#endif
