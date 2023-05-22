"""
@Author: Conghao Wong
@Date: 2023-05-19 09:51:56
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-22 20:07:22
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any

from ...base import BaseManager
from ..__splitManager import Clip


class BaseInputManager(BaseManager):
    """
    BaseInputManager
    ---

    Basic class for all `InputManagers`.
    It should be managed by the `AgentManager` object.
    """
    
    TEMP_FILE: str = None
    TEMP_FILES: dict[str, str] = None

    ROOT_DIR: str = None
    INPUT_TYPE: str = None

    def __init__(self, manager: BaseManager, name: str = None):
        super().__init__(manager=manager, name=name)

        self.__clip: Clip = None

    def run(self, clip: Clip, *args, **kwargs) -> list:
        """
        Run all dataset-related operations within this manager,
        including load, preprocess, read or write files, and
        then make train samples on the given clip.
        """
        self.clean()
        self.__clip = clip
        self.init_clip(clip)

        if not self.temp_file_exists:
            self.save(*args, **kwargs)

        return self.load(*args, **kwargs)
    
    @property
    def working_clip(self) -> Clip:
        if not self.__clip:
            raise ValueError(self.__clip)
        
        return self.__clip
    
    @property
    def temp_dir(self) -> str:
        if not (r := self.ROOT_DIR):
            return self.working_clip.temp_dir
        else:
            return os.path.join(self.working_clip.temp_dir, r)
    
    @property
    def temp_file(self) -> str:
        if not self.TEMP_FILE:
            return None
        
        return os.path.join(self.temp_dir, self.TEMP_FILE)
    
    @property
    def temp_files(self) -> dict[str, str]:
        if not self.TEMP_FILES:
            return None
        
        dic = {}
        for key, value in self.TEMP_FILES.items():
            dic[key] = os.path.join(self.temp_dir, value)
        
        return dic
    
    @property
    def temp_file_exists(self) -> bool:
        files = []
        if (t := self.temp_file):
            files.append(t)
        elif (t := self.temp_files):
            files += list(t.values())
        else:
            raise ValueError('Wrong temp file settings!')
        
        exists = True
        for f in files:
            if not os.path.exists(f):
                exists = False
                break
        
        return exists

    def clean(self):
        self.__clip = None

    def init_clip(self, clip: Clip):
        pass

    def save(self, *args, **kwargs) -> Any:
        """
        Load original dataset files.
        """
        raise NotImplementedError
    
    def load(self, *args, **kwargs) -> list:
        """
        Process and sample the processed data to a list of values
        to train or test.
        """
        raise NotImplementedError
    

    
    