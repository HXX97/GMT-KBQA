# @Time    : 2021/5/27 15:28
# @Author  :


import logging
from torch._C import device
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)


 
# Egnlish to Chinese
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
# model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
 
text = "Student accommodation centres, resorts"
translated_text = translation(text, max_length=40)[0]['translation_text']
print(translated_text)
 
# Chinese to English
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translation = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)
 
text = "学生住宿中心,度假屋"
translated_text = translation(text, max_length=40)[0]['translation_text']
print(translated_text)