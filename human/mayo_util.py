import os
import random


DATA_FOLDER = "You must give the data folder"


class FileLoader(object):
    def __init__(self, folder, exclude):
        self.folder = folder
        self.exclude = exclude
        self.load_files()

    def load_files(self):
        self.files = []
        for (dirpath, dirnames, filenames) in os.walk(self.folder):
            self.files.extend([os.path.join(self.folder, fi) for fi in filenames
                               if not fi.startswith(self.exclude)])
        random.shuffle(self.files)

    def next_file(self):
        if not self.files:
            self.load_files()
        return self.files.pop()
