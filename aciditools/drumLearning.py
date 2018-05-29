from .data.sets.generic import Dataset
from .data.sets.audio import DatasetAudio
from .data.sets.midi import DatasetMidi
from .data.sets.timeseries import DatasetTimeSeries
from .data.sets.metadata import importRawLabel
from .data.sets import utils
# Add extraneous imports 
from .asynchronous.task import AsynchronousTask
#import gc


def importDataset(base_path = '/fast-1/DrumsDataset', transform = 'nsgt-cqt', targetDur = None):
    audioOptions = {
      "dataDirectory":base_path+'/data',              
      "dataPrefix": base_path, 
      "metadataDirectory":base_path + '/metadata',                        
      "analysisDirectory":base_path + '/analysis',# Root to place (and find) the transformed data
      "importCallback":None,                                  # Function to perform import of data
      "transformCallback":None,                               # Function to transform data (can be a list)
      "transformName":transform,                            # Name of the imported transform
      "tasks":['instrument'],                                      # Tasks to import
      "taskCallback":None,                                    # Function to import task metadata
      "verbose":True,                                         # Be verbose or not
      "checkIntegrity":True,                                  # Check that files exist (while loading)
      "forceUpdate":True,                                    # Force the update
      "forceRecompute":False
    };
    
    audioSet = DatasetAudio(audioOptions);
    audioSet.listDirectory();
    
    print('[Import metadata]');
    audioSet.importMetadataTasks();
    
    print('[Compute transforms]')
    transformList, transformParameters= audioSet.getTransforms();
    transformOptions = dict(audioOptions)
    transformOptions["transformTypes"] = ['nsgt-cqt', 'raw']
    transformOptions["transformNames"] = ['nsgt-cqt', 'raw']
    transformOptions['forceRecompute'] = False
    
    transformParameters = [transformParameters, dict(transformParameters)]
    if targetDur is not None:
        transformParameters[0]['targetDuration'] = targetDur
        
    transformParameters[1]['grainSize'] = int(22050*0.1)
    transformParameters[1]['grainHop'] = int(0.25 * 22050*0.2)
    
    transformOptions['transformParameters'] = transformParameters
    audioSet.computeTransforms(None, transformOptions, padding=False)
    
    print('[Import audio]');
    audioOptions['transformOptions'] = transformOptions
    audioSet.importData(None, audioOptions);
    
    return audioSet

#caca