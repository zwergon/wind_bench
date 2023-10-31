import boto3
from wb.dataset.s3.config import Config
from pyarrow import fs


class S3:

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.s3 =  None

    def __enter__(self):
        self.s3 =   boto3.resource('s3',
            endpoint_url=self.config.S3_URL,
            aws_access_key_id=self.config.S3_ACCESS_KEY,
            aws_secret_access_key=self.config.S3_SECRET_KEY,
            aws_session_token=None,
            config=boto3.session.Config(signature_version='s3v4'),
            verify=False
        )
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.s3.meta.client.close()
        

    @property
    def bucket(self):
        return self.s3.Bucket(self.config.S3_BUCKET)
    
    @property
    def filesystem(self):
        return fs.S3FileSystem(
            endpoint_override=self.config.S3_URL,
            access_key= self.config.S3_ACCESS_KEY,
            secret_key= self.config.S3_SECRET_KEY
            )
