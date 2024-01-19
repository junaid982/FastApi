# from pydantic import BaseModel


# class DeBlurConfig(BaseModel):
#     norm_layer:str = 'instance'
#     weight_path = r'models\weight\fpn_inception.h5'


from pydantic import BaseModel
from typing import ClassVar

class DeBlurConfig(BaseModel):
    norm_layer: str = 'instance'
    weight_path: ClassVar[str] = r'models/deblur_gan/weight/fpn_inception.h5'