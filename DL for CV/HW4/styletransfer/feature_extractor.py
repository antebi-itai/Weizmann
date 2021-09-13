import torch
from torchvision import models

##########################################################
# Feature Extractor
##########################################################

class FeatureExtractor:
  """This class facilitates extracting features from a pretrained network for
   style transfer. We recommend you use VGG19 like in the original paper, 
   but you are more than welcome to try other models as well."""

  CONTENT_KEY = 'content'
  STYLE_KEY = 'style'
  KEY_LIST = [CONTENT_KEY, STYLE_KEY]

  def __init__(self, device='cpu', model_type="vgg19"):
    """
    Args:            
      device (string): Device to host model on. Defalutly host on cpu.
      model_type (string): Type of model to extract. Defaultly use vgg19
      pretrained on ImageNet.  
    """
    # initiate a dictionary where every key in KEY_LIST has an empty list as 
    # a value. It will later contain the layers to extract for each key.
    self.layers_dict = {}
    # initiate a dictionary where every key in KEY_LIST has an empty list as 
    # a value. It will later contain the features extracted for each key.
    self.outputs_dict = {}
    for key in FeatureExtractor.KEY_LIST:
      self.layers_dict[key] = []
      self.outputs_dict[key] = []
    # initiate a list to hold the hook handlers.
    self.hook_handlers = []

    # initiate the extractor model
    # BEGIN SOLUTION
    self.model = getattr(models.vgg, model_type)(pretrained=True).to(device)
    # END SOLUTION

  def _get_content_hook(self):
    """
    Defines the hook from extracting a content key.
    Returns:            
      The method to hook. 
    """
    def _get_content_output(model, input, output):
      self.outputs_dict[FeatureExtractor.CONTENT_KEY].append(output)
    return _get_content_output

  def _get_style_hook(self):
    """
    Defines the hook from extracting a style key.
    Returns:            
      The method to hook. 
    """
    def _get_style_output(model, input, output):
      self.outputs_dict[FeatureExtractor.STYLE_KEY].append(output)
    return _get_style_output

  def _register_hooks(self, **kwargs):
    """
    Registers all the hooks to perform extraction.
    Args:            
      kwargs (dict(string, List(int))): dictionary with all keys in KEY_LIST.
      Each key's value is a list of all the layers to extract for this key.
    """
    # BEGIN SOLUTION
    if kwargs is not None:
      self.layers_dict = kwargs
    
    for key in self.layers_dict.keys():
      # find hook func for this key
      if key == FeatureExtractor.CONTENT_KEY:
        hook_func = self._get_content_hook()
      elif key == FeatureExtractor.STYLE_KEY:
        hook_func = self._get_style_hook()
      else:
        raise NotImplementedError("Only supporting content / style features")
      # hook all relevant layers
      for layer_index in self.layers_dict[key]:
        hook_handle = self.model.features[layer_index].register_forward_hook(hook_func)
        self.hook_handlers.append(hook_handle)  
    # END SOLUTION

  def _unregister_hooks(self):
    """
    Unregisters all the hooks after performing an extraction.
    """
    # BEGIN SOLUTION
    for key in FeatureExtractor.KEY_LIST:
      self.layers_dict[key] = []
    
    for hook_handle in self.hook_handlers:
      hook_handle.remove()
    # END SOLUTION
 
  def extract(self, batch, **kwargs):
    """
    Defines the hook from extracting a content key.
    Args:
    batch (Torch.Tensor) A tensor images to extract.
    Has shape `(B, C, H, W)`. 
    kwargs (dict(string, List(int))): dictionary with all keys in KEY_LIST.
    Each key's value is a list of all the layers to extract for this key.
    Returns:            
      A dictionary with keys in KEY_LIST. Every key's value is a list of
      extracted features, according to the requested layers for each key. 
    """
    # BEGIN SOLUTION
    self._register_hooks(**kwargs)
    self.model(batch)
    self._unregister_hooks()
    # save dict to return
    outputs_dict = dict(self.outputs_dict)
    # clean outputs_dict for future reference
    self.outputs_dict = {}
    for key in FeatureExtractor.KEY_LIST:
      self.outputs_dict[key] = []
    
    return outputs_dict
    # END SOLUTION




