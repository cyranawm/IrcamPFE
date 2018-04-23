from data.sets.generic import Dataset
from data.sets.audio import DatasetAudio
from data.sets.midi import DatasetMidi
from data.sets.timeseries import DatasetTimeSeries
from data.sets.metadata import importRawLabel
from data.sets import utils
# Add extraneous imports 
from asynchronous.task import AsynchronousTask
import gc


def importDataset():
    audioOptions = {
      "dataDirectory":'/fast-1/DrumsDataset/data',              
      "dataPrefix":'/fast-1/DrumsDataset', 
      "analysisDirectory":'/fast-1/DrumsDataset/analysis',# Root to place (and find) the transformed data
      "importCallback":None,                                  # Function to perform import of data
      "transformCallback":None,                               # Function to transform data (can be a list)
      "transformName":'nsgt-cqt',                            # Name of the imported transform
      "tasks":['instrument'],                                      # Tasks to import
      "taskCallback":None,                                    # Function to import task metadata
      "verbose":True,                                         # Be verbose or not
      "checkIntegrity":True,                                  # Check that files exist (while loading)
      "forceUpdate":False,                                    # Force the update
      "forceRecompute":False
    };
    
    audioSet = DatasetAudio(audioOptions);
    audioSet.listDirectory();
    
    print('[Import metadata]');
    audioSet.importMetadataTasks();
    
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

