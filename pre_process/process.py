import os
import pickle
import csv
import numpy as np
import random
import torch
from transformers import BertTokenizer
from scale import get_scaled_down_scores
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from encode import encode_documents

@dataclass
class EssayFeatures:
    essay_id: int
    prompt_id: int
    content_text: str
    attributes: List[int]
    handcraft_features: List[float]
    readability_features: List[float]

class EssayAttributeProcessor:
    
    ATTRIBUTES = [
        'score', 
        'content', 
        'organization', 
        'word_choice', 
        'sentence_fluency', 
        'conventions', 
        'prompt_adherence',
        'language', 
        'narrativity']
    
    def __init__(self):
        self.num_attributes = len(self.ATTRIBUTES)
    
    def extract_attributes(self, essay_data):
        attributes = [-1] * self.num_attributes
        
        for attribute_name in essay_data.keys():
            if attribute_name in self.ATTRIBUTES:
                attribute_index = self.ATTRIBUTES.index(attribute_name)
                attributes[attribute_index] = int(essay_data[attribute_name])
        
        return attributes

class TextEncoder:
    def __init__(self, tokenizer_name = 'bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.chunk_sizes = [90, 30, 130, 10]
    
    def encode_single_document(self, texts, max_length = 512):
        inputs_single, *_ = encode_documents(texts, self.tokenizer, max_input_length=max_length)
        return inputs_single
    
    def encode_chunked_documents(self, texts) -> List[Any]:
        chunked_inputs = []
        
        for chunk_size in self.chunk_sizes:
            encoded_chunk, *_ = encode_documents(texts, self.tokenizer, max_input_length=chunk_size)
            chunked_inputs.append(encoded_chunk)
        
        return chunked_inputs



class BertInputProcessor:
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102

    def process_bert_inputs(self, chunked_inputs) :

        processed_batches = []
        valid_lengths = []

        for batch in chunked_inputs:
            # batch: (B, S, 3, L) B = nums_essays S = num_segments
            B, S, _, L = batch.shape
            proc = batch.clone()         
            valid_lens_for_batch = []

            for i in range(B):           
                valid_count = 0
                for j in range(S):       
                    seg = proc[i, j]     
                    input_ids   = seg[0]
                    token_types = seg[1]
                    masks       = seg[2]

                    if not torch.all(input_ids == 0):
                        valid_count += 1
                    else:
                        input_ids[0]  = self.CLS_TOKEN_ID
                        input_ids[1]  = self.SEP_TOKEN_ID
                        token_types.zero_()
                        masks[:2]    = 1
                        proc[i, j]   = torch.stack((input_ids, token_types, masks))

                valid_lens_for_batch.append(valid_count)

            processed_batches.append(proc)
            valid_lengths.append(valid_lens_for_batch)

        return processed_batches, valid_lengths



class FeatureDataLoader:

    def __init__(self, handcraft_path: str, readability_path: str):
        self.handcraft_features = self.load_handcraft_features(handcraft_path)
        self.readability_features = self.load_readability_features(readability_path)
    
    def load_handcraft_features(self, path):
        with open(path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            return list(reader)
    
    def load_readability_features(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def get_handcraft_features(self, essay_id):
        for feature in self.handcraft_features:
            if feature[0] == essay_id:
                return feature[2:]
        return []
    
    def get_readability_features(self, essay_id):
        for feature in self.readability_features:
            if int(feature[0]) == essay_id:
                return list(feature[1:])
        return []

class EssayDataProcessor:

    def __init__(self, handcraft_path, readability_path):
        self.attribute_processor = EssayAttributeProcessor()
        self.text_encoder = TextEncoder()
        self.bert_processor = BertInputProcessor()
        self.feature_loader = FeatureDataLoader(handcraft_path, readability_path)
    
    def process_essay_data(self, raw_data):
        processed_essays = []
        
        for essay_data in raw_data:
            essay_features = self.create_essay_features(essay_data)
            processed_essays.append(essay_features)
        
        return processed_essays
    
    def create_essay_features(self, essay_data):
        essay_id = int(essay_data['essay_id'])
        prompt_id = int(essay_data['prompt_id'])
        content_text = essay_data['content_text']
        
        attributes = self.attribute_processor.extract_attributes(essay_data)
        handcraft_features = self.feature_loader.get_handcraft_features(str(essay_id))
        readability_features = self.feature_loader.get_readability_features(essay_id)
        
        return EssayFeatures(
            essay_id=essay_id,
            prompt_id=prompt_id,
            content_text=content_text,
            attributes=attributes,
            handcraft_features=handcraft_features,
            readability_features=readability_features
        )
    
    def encode_and_process(self, essays):
        texts = [essay.content_text for essay in essays]
        labels = [essay.attributes for essay in essays]
        prompt_ids = [essay.prompt_id for essay in essays]
        
        inputs_single = self.text_encoder.encode_single_document(texts)
        inputs_chunked = self.text_encoder.encode_chunked_documents(texts)
        
        processed_inputs, valid_lengths = self.bert_processor.process_bert_inputs(inputs_chunked)
        scaled_scores = get_scaled_down_scores(labels, prompt_ids)
        
        return {
            'inputs_single': inputs_single,
            'inputs_chunked': processed_inputs,
            'scaled_score': scaled_scores,
            'prompt_id': prompt_ids,
            'valid_length': valid_lengths,
            'hand_craft': [essay.handcraft_features for essay in essays],
            'readability': [essay.readability_features for essay in essays],
            'essay_id': [essay.essay_id for essay in essays]
        }

    def encode_and_process_prompt(self, essays):
        texts = [essay.content_text for essay in essays]
        prompt_ids = [essay.prompt_id for essay in essays]
        
        inputs_single = self.text_encoder.encode_single_document(texts)
        inputs_chunked = self.text_encoder.encode_chunked_documents(texts)
        
        processed_inputs, valid_lengths = self.bert_processor.process_bert_inputs(inputs_chunked)
        
        return {
            'inputs_single': inputs_single,
            'inputs_chunked': processed_inputs,
            'prompt_id': prompt_ids,
            'valid_length': valid_lengths,
            'hand_craft': [essay.handcraft_features for essay in essays],
            'readability': [essay.readability_features for essay in essays],
        }

        
    def process_prompts(self, texts, handcraft_path, readability_path):
        prompt_feature_loader = FeatureDataLoader(handcraft_path, readability_path)
        
        prompt_essays = []
        for i, text in enumerate(texts, 1):
            handcraft_features = prompt_feature_loader.get_handcraft_features(str(i))
            readability_features = prompt_feature_loader.get_readability_features(i)
            
            essay = EssayFeatures(
                essay_id=None,
                prompt_id=i,
                content_text=text,
                attributes=None,
                handcraft_features=handcraft_features,
                readability_features=readability_features
            )
            prompt_essays.append(essay)
        
        result = self.encode_and_process_prompt(prompt_essays)
           
        return result

class DatasetManager:
    
    def __init__(self, base_path, handcraft_path, readability_path):
        self.base_path = base_path
        self.processor = EssayDataProcessor(handcraft_path, readability_path)
    
    def process_and_save_dataset(self, prompt_number, split, output_path):
        data = self.load_pickle_data(
            os.path.join(self.base_path, str(prompt_number), f"{split}.pk")
        )

        processed_essays = self.processor.process_essay_data(data)
        result = self.processor.encode_and_process(processed_essays)

        os.makedirs(output_path, exist_ok=True)
        fname = f"encode_prompt_{prompt_number}.pkl"
        with open(os.path.join(output_path, fname), 'wb') as f:
            pickle.dump(result, f)

    def process_and_save_prompts(self, prompt_texts, prompt_out_path, prompt_handcraft_path, prompt_readability_path):
        result = self.processor.process_prompts(
            prompt_texts, 
            prompt_handcraft_path, 
            prompt_readability_path
        )

        #os.makedirs(os.path.dirname(prompt_out_path), exist_ok=True)
        with open(prompt_out_path, "wb") as f:
            pickle.dump(result, f) 
        print(f"prompt encoded") 
        
    def load_pickle_data(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    
        
if __name__ == "__main__":
    base_path = "cross_prompt_attributes"
    handcraft_path = "var_norm.csv"
    readability_path = "var_norm_readability.pickle"

    prompt_handcraft_path = "var_prompt_norm.csv"
    prompt_readability_path = "prompt_readability.pickle"
    prompt_out_path = "new_encode_prompts.pkl"
    prompt_path = "prompt.txt"
    
    dataset_manager = DatasetManager(base_path, handcraft_path, readability_path)

    with open(prompt_path, encoding='utf-8') as f:
        prompt_list = [line.strip() for line in f if line.strip()]
    dataset_manager.process_and_save_prompts(
        prompt_list,
        prompt_out_path,
        prompt_handcraft_path,
        prompt_readability_path
    )
    
    process = ["train", "dev", "test"]
    for prompt_number in tqdm(range(1, 9)):
        for split in process:
            output_path = f"dataset/new_{split}"
            dataset_manager.process_and_save_dataset(prompt_number, split, output_path)

            

        