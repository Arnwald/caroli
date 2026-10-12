/* Serial MPI stub (one rank) for the thesis reference codes.
 *
 * Implements the Fortran MPI calls used by the thesis codes with a single process:
 * collectives become copies, broadcasts are no-ops, point-to-point calls abort.
 * Datatype handles from mpif.h encode the element size: size = handle / 100.
 * Symbols follow gfortran's convention (lower case, trailing underscore,
 * hidden string lengths appended as size_t).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static size_t type_size(int datatype) { return (size_t)(datatype / 100); }

static void copy(void *dst, const void *src, size_t nbytes) {
    if (dst != src && nbytes > 0) memmove(dst, src, nbytes);
}

static void unsupported(const char *name) {
    fprintf(stderr, "mpistub: %s is not supported with a single rank\n", name);
    exit(3);
}

void mpi_init_(int *ierr) { *ierr = 0; }
void mpi_finalize_(int *ierr) { *ierr = 0; }
void mpi_barrier_(int *comm, int *ierr) { (void)comm; *ierr = 0; }

void mpi_comm_rank_(int *comm, int *rank, int *ierr) { (void)comm; *rank = 0; *ierr = 0; }
void mpi_comm_size_(int *comm, int *size, int *ierr) { (void)comm; *size = 1; *ierr = 0; }

void mpi_get_processor_name_(char *name, int *len, int *ierr, size_t name_len) {
    const char *host = "serial-stub";
    size_t n = strlen(host);
    if (n > name_len) n = name_len;
    memcpy(name, host, n);
    memset(name + n, ' ', name_len - n);
    *len = (int)n;
    *ierr = 0;
}

void mpi_bcast_(void *buf, int *count, int *datatype, int *root, int *comm, int *ierr) {
    (void)buf; (void)count; (void)datatype; (void)root; (void)comm;
    *ierr = 0;
}

void mpi_reduce_(void *sendbuf, void *recvbuf, int *count, int *datatype, int *op, int *root,
                 int *comm, int *ierr) {
    (void)op; (void)root; (void)comm;
    copy(recvbuf, sendbuf, (size_t)*count * type_size(*datatype));
    *ierr = 0;
}

void mpi_allreduce_(void *sendbuf, void *recvbuf, int *count, int *datatype, int *op, int *comm,
                    int *ierr) {
    (void)op; (void)comm;
    copy(recvbuf, sendbuf, (size_t)*count * type_size(*datatype));
    *ierr = 0;
}

void mpi_gather_(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcount,
                 int *recvtype, int *root, int *comm, int *ierr) {
    (void)recvcount; (void)recvtype; (void)root; (void)comm;
    copy(recvbuf, sendbuf, (size_t)*sendcount * type_size(*sendtype));
    *ierr = 0;
}

void mpi_gatherv_(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcounts,
                  int *displs, int *recvtype, int *root, int *comm, int *ierr) {
    (void)recvcounts; (void)root; (void)comm;
    copy((char *)recvbuf + (size_t)displs[0] * type_size(*recvtype), sendbuf,
         (size_t)*sendcount * type_size(*sendtype));
    *ierr = 0;
}

void mpi_scatter_(void *sendbuf, int *sendcount, int *sendtype, void *recvbuf, int *recvcount,
                  int *recvtype, int *root, int *comm, int *ierr) {
    (void)sendcount; (void)sendtype; (void)root; (void)comm;
    copy(recvbuf, sendbuf, (size_t)*recvcount * type_size(*recvtype));
    *ierr = 0;
}

void mpi_scatterv_(void *sendbuf, int *sendcounts, int *displs, int *sendtype, void *recvbuf,
                   int *recvcount, int *recvtype, int *root, int *comm, int *ierr) {
    (void)root; (void)comm;
    int n = sendcounts[0] < *recvcount ? sendcounts[0] : *recvcount;
    copy(recvbuf, (char *)sendbuf + (size_t)displs[0] * type_size(*sendtype),
         (size_t)n * type_size(*recvtype));
    *ierr = 0;
}

void mpi_send_(void) { unsupported("MPI_Send"); }
void mpi_recv_(void) { unsupported("MPI_Recv"); }

void mpi_abort_(int *comm, int *errorcode, int *ierr) {
    (void)comm; (void)ierr;
    exit(*errorcode);
}

double mpi_wtime_(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
