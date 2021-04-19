import os
import sys

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


# upload files or folder to drive
class google_drive:

    def __init__(self):
        self.folder_path = None
        self.file_paths = None

    def upload_files(self, target_parent_id, folder_path=None, file_paths=None):
        gauth = GoogleAuth()
        drive = GoogleDrive(gauth)
        # if gauth.credentials is None:
        #     # Authenticate if they're not there
        #     gauth.LocalWebserverAuth()
        # elif gauth.access_token_expired:
        #     # Refresh them if expired
        #     gauth.Refresh()
        # else:
        #     # Initialize the saved creds
        #     gauth.Authorize()
        self.folder_path = folder_path
        self.file_paths = file_paths

        if self.folder_path is not None:
            self.file_paths = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if
                               os.path.isfile(os.path.join(self.folder_path, f))]
        elif self.file_paths is None:
            return
        for file_path in self.file_paths:
            gfile = drive.CreateFile({'parents': [{'id': target_parent_id}], 'title': os.path.basename(file_path)})
            gfile.SetContentFile(file_path)
            gfile.Upload()  # Upload the file.


if __name__ == '__main__':
    folder_name = '../../attacker/saved_images/l2'
    args = sys.argv
    drive = google_drive()

    if args[1] == 'l0':
        drive.upload_files(target_parent_id='1z3RFnOdUH8OA8xyqQfaUnqY7qnY68TeY',
                           folder_path="../../attacker/saved_images/l0")

    elif args[2] == 'l2':
        drive.upload_files(target_parent_id='1uB-HX80YMQIpj1NMUmEdwWwA0hJUYW7q', folder_path=folder_name)
    else:
        print('cannot upload !')
