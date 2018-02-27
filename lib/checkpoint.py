import logging
import os
import time

import dill
import torch

import lib.utils

logger = logging.getLogger(__name__)
import_time = time.time()


class Checkpoint(object):

    def __init__(self, checkpoint_path, device=None):
        """
        Load a checkpoint.

        Args:
            device (int)
            checkpoint_path (str or None): Given a non-none checkpoint path, the checkpoint is
                loaded
        """
        self.device = lib.utils.device_default(device)
        self.checkpoint_path = checkpoint_path

        logger.info("Loading checkpoints from %s onto device %d", self.checkpoint_path, self.device)

        # http://pytorch.org/docs/master/torch.html?highlight=torch%20load#torch.load
        def remap(storage, loc):
            if 'cuda' in loc and self.device >= 0:
                return storage.cuda(device=self.device)
            return storage

        data = torch.load(self.checkpoint_path, map_location=remap, pickle_module=dill)
        # https://stackoverflow.com/questions/2535917/copy-kwargs-to-self
        for (k, v) in data.items():
            setattr(self, k, v)

    @classmethod
    def recent(cls, log_directory, device=None):
        """
        Load a checkpoint or returns `None` if log_directory has no checkpoint.

        Args:
            log_directory (str or None): Lastest checkpoint is loaded from log_directory
            device (int)
        """
        all_filenames = sorted(os.listdir(log_directory), reverse=True)
        all_checkpoints = [filename for filename in all_filenames if '.pt' in filename]
        if len(all_checkpoints) == 0:
            return None
        checkpoint_path = os.path.join(log_directory, all_checkpoints[0])
        return cls(checkpoint_path, device)

    @classmethod
    def save(cls, folder, data, device=None):
        """
        Saves the current model and related training parameters into a subdirectory of the
        checkpoint directory.

        Args:
            folder (str): path to the save directory
            object (dict): object to save
            device (int): give a device number to be appended to the end of the path
        """
        time_elapsed = int(time.time() - import_time)
        name = '%d.%d' % (time_elapsed, device) if device else str(time_elapsed)
        name += '.pt'
        path = os.path.join(folder, name)

        if os.path.exists(path):
            logger.error('Cannot save checkpoint; path (%s) already exists.', path)
            return

        logger.info('Saving checkpoint: %s', path)

        torch.save(data, path, pickle_module=dill)

        return path
