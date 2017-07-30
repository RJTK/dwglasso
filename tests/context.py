'''A file for setting up the proper import environment for test files.  Running

from .context import dwglasso

in a test file will import ../dwglasso into the test.
'''
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


import dwglasso
