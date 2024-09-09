import os

import boto3
from botocore.exceptions import NoCredentialsError

def download_folder_from_s3(bucket_name, folder_key, local_folder_path):
    """
    Downloads a file from an Amazon S3 bucket using the boto3 library.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - folder_key (str): The key or path of the file/folder in the S3 bucket.
    - local_folder_name (str): The desired name for the downloaded folder on the local machine.

    Returns:
    None

    Note:
    - The function uses hardcoded AWS access key ID and secret access key for authentication,
      which is not recommended for security reasons.
    - It initializes a boto3 S3 client with the provided access credentials and attempts to download
      the specified file from the S3 bucket.
    - If successful, it prints a confirmation message with details of the download, including the folder key,
      bucket name, and local folder name.
    - If an exception of type 'NoCredentialsError' occurs, it prints a message indicating that credentials
      are not available.

    Recommendation:
    - Using hardcoded credentials is not a secure practice. It is recommended to use IAM roles or
      environment variables for better security.
    """
    aws_access_key_id = 'xxxxxxxxxxxxxxxx'
    aws_secret_access_key ='xxxxxxxxxxxxxxxx//'

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    try:
        # List all objects in the specified S3 folder recursively
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_key)

        for obj in objects.get('Contents', []):
            file_key = obj['Key']
            local_file_path = os.path.join(local_folder_path, file_key[len(folder_key):])

            # Ensure local directories exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            print(f"Started downloading {file_key} from {bucket_name}")
            s3.download_file(bucket_name, file_key, local_file_path)
            print(f"Finished downloading {file_key} from {bucket_name} as: {local_file_path}")

    except NoCredentialsError:
        print("Credentials not available")

# Fetch the data set.
bucket_name = 'capstone-efficient-waste-sorting-202402'
# TODO: Change this to the appropriate data set names.
folder_key = 'npy/'
local_folder_path = "npy/"

download_folder_from_s3(bucket_name, folder_key, local_folder_path)
