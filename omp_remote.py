"""Start isolated OMP with the remote inference server."""

from omp_local import main


if __name__ == "__main__":
    main(remote=True)
