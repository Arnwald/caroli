/* Fortran (legacy f77 interface) wrappers for the FFTW threads API.
 *
 * Some FFTW builds, such as Homebrew's, ship libfftw3_threads without the Fortran wrappers
 * dfftw_init_threads, dfftw_plan_with_nthreads and dfftw_cleanup_threads used by fft_mod.f90.
 */
#include <fftw3.h>

void dfftw_init_threads_(int *iret) { *iret = fftw_init_threads(); }
void dfftw_plan_with_nthreads_(int *nthreads) { fftw_plan_with_nthreads(*nthreads); }
void dfftw_cleanup_threads_(void) { fftw_cleanup_threads(); }
