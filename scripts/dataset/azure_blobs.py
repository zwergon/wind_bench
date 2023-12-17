from io import BytesIO
import pyarrow.parquet as pq

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# <Snippet_list_containers>
def list_containers(blob_service_client: BlobServiceClient):
    containers = blob_service_client.list_containers(include_metadata=True)
    for container in containers:
        print(container["name"], container["metadata"])


# </Snippet_list_containers>


def list_containers_pages(blob_service_client: BlobServiceClient):
    i = 0
    all_pages = blob_service_client.list_containers(results_per_page=5).by_page()
    for container_page in all_pages:
        i += 1
        print(f"Page {i}")
        for container in container_page:
            print(container["name"])


def list_blobs_flat(blob_service_client: BlobServiceClient, container_name):
    container_client = blob_service_client.get_container_client(
        container=container_name
    )

    blob_list = container_client.list_blobs()

    blobs = []

    for blob in blob_list:
        if ".parquet" in blob.name:
            blobs.append(blob.name)
            # print(f"Name: {blob.name}")

    return blobs


def get_parquet(blob_service_client: BlobServiceClient, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name
    )
    byte_stream = BytesIO()

    blob_client.download_blob().readinto(byte_stream)
    df = pq.read_table(byte_stream, columns=["Rotor azimuth [deg]"]).to_pandas()
    return df


if __name__ == "__main__":
    account_url = "https://mlparquetstorage.blob.core.windows.net"
    credential = DefaultAzureCredential()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(
        account_url,
        credential=credential,
        proxies={"http:": "http://irproxy:8082", "https": "http://irproxy:8082"},
    )
    list_containers(blob_service_client)
    blobs = list_blobs_flat(blob_service_client, "windbench")
    print(blobs[0])

    df = get_parquet(blob_service_client, "windbench", blobs[0])
    print(df.head())
