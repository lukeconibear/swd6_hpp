import dask.array as da
from dask_mpi import initialize
from dask.distributed import Client, performance_report

initialize()
client = Client()

def example_function():
    """
    Example linear algebra with Dask Array to demonstrate diagnostics.
    Taken from https://docs.dask.org/en/stable/diagnostics-local.html#example
    """
    random_array = da.random.random(size=(10_000, 1_000), chunks=(1_000, 1_000))
    # take the QR decomposition: https://en.wikipedia.org/wiki/QR_decomposition
    q, r = da.linalg.qr(random_array)
    random_array_reconstructed = q.dot(r)

    with performance_report(filename="dask-report_example_mpi_sge.html"):
        result = random_array_reconstructed.compute()


example_function()
