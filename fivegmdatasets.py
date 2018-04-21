import json
import os
import tempfile
import io
import zipfile

import requests

from rwisimulation.datamodel import save5gmdata as fgdb

from pathlib import Path
home = str(Path.home())
user_conf_path = os.path.join(home, '.config', '.5gmdata.json')

try:
    with open(user_conf_path) as f:
        config = json.load(f)
except FileNotFoundError:
    config = dict()
    config['urban_cannyon_v2i.5gmv1'] = tempfile.mkdtemp()
with open(user_conf_path, 'w') as f:
    json.dump(config, f)

def _download_file(url, output_file):
    # https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
    r = requests.get(url)
    for chunk in r.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            output_file.write(chunk)


def urban_cannyon_v2i(cached=True):
    urban_cannyon_v2i_file = open(
        os.path.join(config['urban_cannyon_v2i.5gmv1'], 'archive.zip'),
        'ab')
    if cached:
        urban_cannyon_v2i_file.seek(0, io.SEEK_END)
        if urban_cannyon_v2i_file.tell() == 0:
            _download_file('https://owncloud.lasseufpa.org/s/wLCtqEYwHiSqMU1/download', urban_cannyon_v2i_file)
    else:
        _download_file('https://owncloud.lasseufpa.org/s/wLCtqEYwHiSqMU1/download', urban_cannyon_v2i_file)
    urban_zip = zipfile.ZipFile(urban_cannyon_v2i_file.name)
    urban_zip.extract('2018-03-18-urban-cannyon-v2i-5gmv1/urban_cannyon_v2i.5gmv1', config['urban_cannyon_v2i.5gmv1'])
    engine = fgdb.create_engine('sqlite:////' + os.path.join(config['urban_cannyon_v2i.5gmv1'],
                                                             '2018-03-18-urban-cannyon-v2i-5gmv1/urban_cannyon_v2i.5gmv1'))
    Session = fgdb.sessionmaker(bind=engine)
    Session.configure(bind=engine)
    return Session()
    #session = fgdb.Session()


if __name__ == '__main__':
    session = urban_cannyon_v2i()
    print(session.query(fgdb.Episode).count())