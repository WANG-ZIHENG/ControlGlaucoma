# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from typing import Union, List, Tuple, Optional
import logging

from vit_seg_modeling import VisionTransformer as ViT_seg
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class TransUNetInterface:
    
    def __init__(self, 
                 model_path: str,
                 vit_name: str = 'R50-ViT-B_16',
                 num_classes: int = 3,
                 img_size: int = 224,
                 n_skip: int = 3,
                 vit_patches_size: int = 16,
                 device: str = 'auto'):
        self.model_path = model_path
        self.vit_name = vit_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.n_skip = n_skip
        self.vit_patches_size = vit_patches_size
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self._load_model()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        try:
            config_vit = CONFIGS_ViT_seg[self.vit_name]
            config_vit.n_classes = self.num_classes
            config_vit.n_skip = self.n_skip
            config_vit.patches.size = (self.vit_patches_size, self.vit_patches_size)
            
            if self.vit_name.find('R50') != -1:
                config_vit.patches.grid = (int(self.img_size / self.vit_patches_size), 
                                         int(self.img_size / self.vit_patches_size))
            
            self.model = ViT_seg(config_vit, img_size=self.img_size, num_classes=config_vit.n_classes)
            
            if hasattr(config_vit, 'pretrained_path') and config_vit.pretrained_path:
                if os.path.exists(config_vit.pretrained_path):
                    self.model.load_from(weights=np.load(config_vit.pretrained_path))

            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise e
    

    
    def get_model_info(self) -> dict:
        return {
            'model_path': self.model_path,
            'vit_name': self.vit_name,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'n_skip': self.n_skip,
            'vit_patches_size': self.vit_patches_size,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }




