! Serial MPI stub (one rank) for building the thesis reference codes without an MPI library.
! Datatype handles encode the element size in bytes: size = handle / 100 (see mpistub.c).
      integer, parameter :: MPI_COMM_WORLD = 91
      integer, parameter :: MPI_SUCCESS = 0
      integer, parameter :: MPI_MAX_PROCESSOR_NAME = 256
      integer, parameter :: MPI_STATUS_SIZE = 6
      integer, parameter :: MPI_ANY_SOURCE = -1, MPI_ANY_TAG = -1
      integer, parameter :: MPI_SUM = 1, MPI_MAX = 2, MPI_MIN = 3, MPI_PROD = 4
      integer, parameter :: MPI_CHARACTER = 101
      integer, parameter :: MPI_INTEGER = 401, MPI_REAL = 402, MPI_LOGICAL = 403
      integer, parameter :: MPI_DOUBLE_PRECISION = 801, MPI_DOUBLE = 802, MPI_REAL8 = 803
      integer, parameter :: MPI_COMPLEX = 804, MPI_INTEGER8 = 805
      integer, parameter :: MPI_DOUBLE_COMPLEX = 1601, MPI_COMPLEX16 = 1602
      integer, parameter :: MPI_STATUS_IGNORE(MPI_STATUS_SIZE) = 0
      double precision MPI_WTIME
      external MPI_WTIME
