from pydantic import BaseModel


class DataConfig(BaseModel):
    dataset_name: str = 'bentrevett/multi30k'
    src_lang: str = 'en'
    tgt_lang: str = 'de'
