from aciditools.data.sets.generic import Dataset
from aciditools.data.sets.audio import DatasetAudio
from aciditools.data.sets.midi import DatasetMidi
from aciditools.data.sets.timeseries import DatasetTimeSeries
from aciditools.data.sets.metadata import importRawLabel
from aciditools.data.sets import utils
# Add extraneous imports 
from aciditools.asynchronous.task import AsynchronousTask
import gc


def importDataset():
    audioOptions = {
      "dataDirectory":'/fast-1/DrumsDataset/data',              
      "dataPrefix":'/fast-1/DrumsDataset', 
      "analysisDirectory":'/fast-1/DrumsDataset/analysis',# Root to place (and find) the transformed data
      "importCallback":None,                                  # Function to perform import of data
      "transformCallback":None,                               # Function to transform data (can be a list)
      "transformName":'nsgt-cqt',                            # Name of the imported transform
      "tasks":['genre'],                                      # Tasks to import
      "taskCallback":None,                                    # Function to import task metadata
      "verbose":True,                                         # Be verbose or not
      "checkIntegrity":True,                                  # Check that files exist (while loading)
      "forceUpdate":False,                                    # Force the update
      "forceRecompute":False
    };
    
    audioSet = DatasetAudio(audioOptions);
    audioSet.listDirectory();
    
    print('[Compute transforms]')
    transformList, transformParameters= audioSet.getTransforms();
    transformOptions = dict(audioOptions)
    transformOptions["transformTypes"] = ['nsgt-cqt']
    transformOptions["transformNames"] = ['nsgt-cqt']
    transformOptions['forceRecompute'] = False
    
    transformParameters = [transformParameters]
    transformParameters[0]['targetDuration'] = 1
    
    transformOptions['transformParameters'] = transformParameters
    audioSet.computeTransforms(None, transformOptions, padding=False)
    
    print('[Import audio]');
    audioOptions['transformOptions'] = transformOptions
    audioSet.importData(None, audioOptions);
    
    return audioSet

