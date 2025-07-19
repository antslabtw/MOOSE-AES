# from torch.utils.data import Dataset, DataLoader
# import pickle
# import torch
# import numpy as np

# class CustomDataset(Dataset):
#     def __init__(self, file_path):
#         self.file = file_path
#         with open(file_path, 'rb') as f:
#             self.data = pickle.load(f)
#         with open(f"../feacture/encode_prompt.pickle", "rb") as f:
#             self.prompt = pickle.load(f)
    
#     def __len__(self):
#         return int(len(self.data['inputs_single']))

#     def __getitem__(self, idx):
#         document_single = self.data['inputs_single'][idx]
#         chunked_documents = [chunk[idx] for chunk in self.data['inputs_chunked']]
#         label = np.array(self.data['scaled_score'][idx]).astype(np.float32)
#         prompt_id = self.data['prompt_id'][idx]
#         length = [lengths[idx] for lengths in self.data['valid_length']]
#         readability = torch.tensor([float(x) for x in self.data['readability'][idx]])
#         hand_craft = torch.tensor([float(x) for x in self.data['hand_craft'][idx]])
#         hand_craft = hand_craft[:-1]
#         #----------------------------------------------------------------------------------------------------------------------------------
        
#         prompt_inputs_single = self.prompt['inputs_single'][prompt_id-1]
#         prompt_inputs_chunked = [chunk[prompt_id-1] for chunk in self.prompt['inputs_chunked']]
#         prompt_valid_length = [lengths[prompt_id-1] for lengths in self.prompt['valid_length']]
#         prompt_readability = torch.tensor([float(x) for x in self.prompt['readability'][prompt_id-1]])
#         prompt_hand_craft = torch.tensor([float(x) for x in self.prompt['hand_craft'][prompt_id-1]])
#         prompt_hand_craft = prompt_hand_craft[:-1]

#         return {
#             "prompt_id": prompt_id,
#             "document_single": document_single,
#             "chunked_documents": chunked_documents,
#             "lengths": length,
#             "hand_craft": hand_craft,
#             "readability": readability,
#             "scaled_score": label,
            
#             "prompt_document_single": prompt_inputs_single,
#             "prompt_chunked_documents": prompt_inputs_chunked,
#             "prompt_lengths": prompt_valid_length,
#             "prompt_hand_craft": prompt_hand_craft,
#             "prompt_readability": prompt_readability,
       
#         }
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.file = file_path
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(f"../feacture/encode_prompt.pickle", "rb") as f:
            self.prompt = pickle.load(f)
    
    def __len__(self):
        return int(len(self.data['inputs_single']))

    def __getitem__(self, idx):
        essay_id = self.data['essay_id'][idx]
        document_single = self.data['inputs_single'][idx]
        chunked_documents = [chunk[idx] for chunk in self.data['inputs_chunked']]
        label = np.array(self.data['scaled_score'][idx]).astype(np.float32)
        prompt_id = self.data['prompt_id'][idx]
        length = [lengths[idx] for lengths in self.data['valid_length']]
        readability = torch.tensor([float(x) for x in self.data['readability'][idx]])
        hand_craft = torch.tensor([float(x) for x in self.data['hand_craft'][idx]])
        hand_craft = hand_craft[:-1]
        #----------------------------------------------------------------------------------------------------------------------------------
        
        prompt_inputs_single = self.prompt['inputs_single'][prompt_id-1]
        prompt_inputs_chunked = [chunk[prompt_id-1] for chunk in self.prompt['inputs_chunked']]
        prompt_valid_length = [lengths[prompt_id-1] for lengths in self.prompt['valid_length']]
        prompt_readability = torch.tensor([float(x) for x in self.prompt['readability'][prompt_id-1]])
        prompt_hand_craft = torch.tensor([float(x) for x in self.prompt['hand_craft'][prompt_id-1]])
        prompt_hand_craft = prompt_hand_craft[:-1]

        return {
            "prompt_id": prompt_id,
            "essay_id": essay_id,
            "document_single": document_single,
            "chunked_documents": chunked_documents,
            "lengths": length,
            "hand_craft": hand_craft,
            "readability": readability,
            "scaled_score": label,
            
            "prompt_document_single": prompt_inputs_single,
            "prompt_chunked_documents": prompt_inputs_chunked,
            "prompt_lengths": prompt_valid_length,
            "prompt_hand_craft": prompt_hand_craft,
            "prompt_readability": prompt_readability,
       
        }