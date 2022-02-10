import dask.array as da
from dask_jobqueue import SGECluster
from dask.distributed import Client, performance_report
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize


def setup_client_and_cluster(
    number_processes=1, number_jobs=1, walltime="00:01:00", memory=1
):
    """
    Setup Dask client and cluster.
    Ensure that the number of workers is the right amount for your job and will be fully utilised.
    """
    print("Setting up Dask client and cluster ...")
    number_workers = number_processes * number_jobs
    # these are the requirements for a single worker
    cluster = SGECluster(
        interface="ib0",
        walltime=walltime,
        memory=f"{memory} G",
        resource_spec=f"h_vmem={memory}G",
        scheduler_options={"dashboard_address": ":2727"},
        job_extra=[
            "-V",  # export all environment variables
            f"-pe smp {number_processes}",
            f"-l disk={memory}G"
        ],
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-worker-space"]),
    )
    client = Client(cluster)
    cluster.scale(jobs=number_jobs)
    print("The resources of each worker are: ")
    print(cluster.job_script())
    return client, cluster


def example_function():
    """
    Example linear algebra with Dask Array to demonstrate diagnostics.
    Taken from https://docs.dask.org/en/stable/diagnostics-local.html#example
    """
    random_array = da.random.random(size=(10_000, 1_000), chunks=(1_000, 1_000))
    # take the QR decomposition: https://en.wikipedia.org/wiki/QR_decomposition
    q, r = da.linalg.qr(random_array)
    random_array_reconstructed = q.dot(r)

    with performance_report(filename="dask-report.html"):
        result = random_array_reconstructed.compute()

    np.testing.assert_allclose(random_array, result, rtol=1e-5)


def main():
    client, cluster = setup_client_and_cluster(
        number_processes=1,
        number_jobs=8,
        walltime="00:10:00",
        memory=1,
    )

    print("Main processing ...")
    example_function()
    print("Finished processing.")

    client.close()
    cluster.close()
    print("Closed client and cluster.")


if __name__ == "__main__":
    main()