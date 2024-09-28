from setuptools import setup
import os
import json

setup_args = {
    'name':         'mwa_qa',
    'author':       'Chuneeta Nunhokee',
    'url':          'https://github.com/Chuneeta/mwa_qa',
    'license':      'BSD',
    'version':      "0.0.0-dev",
    'description':  'MWA Data Quality Analysis.',
    'packages':     ['mwa_qa'],
    'package_dir':  {'mwa_qa': 'mwa_qa'},
    'install_requires': [
        'astropy>3.0.0',
        'kneed==0.8.5',
        'matplotlib==3.9.0',
        'numpy==1.24.4',
        'pandas==2.0.3',
        'pytest',
        'python_dateutil>=2.6.0',
        'seaborn',
        'scipy',
    ],
    'include_package_data': True,
    'zip_safe':     False,
    'scripts': ['scripts/run_calqa.py', 'scripts/run_imgqa.py',
                'scripts/run_visqa.py', 'scripts/run_prepvisqa.py',
                'scripts/plot_ants.py', 'scripts/plot_reds.py',
                'scripts/plot_calqa.py', 'scripts/plot_caljson.py',
                'scripts/plot_imgqa.py',
                'scripts/plot_visqa.py', 'scripts/plot_prepvisqa.py',
                'scripts/calqa_to_csv.py', 'scripts/imgqa_to_csv.py',
                'scripts/merge_csvfiles.py', 'scripts/gsheet_to_csv.py',
                'scripts/eval_cutoff_threshold.py', 'scripts/eval_threshold_ps.py',
                'scripts/eval_threshold_img.py']
}

if __name__ == '__main__':
    setup(*(), **setup_args)
