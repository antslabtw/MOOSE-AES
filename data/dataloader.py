from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.file = file_path
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
    def __len__(self):
        return int(len(self.data['inputs_single']))

    def __getitem__(self, idx):
        document_single = self.data['inputs_single'][idx]
        chunked_documents = [chunk[idx] for chunk in self.data['inputs_chunked']]
        label = np.array(self.data['scaled_score'][idx]).astype(np.float32)
        prompt_id = self.data['prompt_id'][idx]
        length = [lengths[idx] for lengths in self.data['valid_length']]
        readability = torch.tensor([float(x) for x in self.data['readability'][idx]])
        #readability = readability[:-1]
        hand_craft = torch.tensor([float(x) for x in self.data['hand_craft'][idx]])
        hand_craft = hand_craft[:-1]
        encode_prompt = self.data['encode_prompt'][idx].float()

        return {
            "prompt_id": prompt_id,
            "document_single": document_single,
            "chunked_documents": chunked_documents,
            "lengths": length,
            "hand_craft": hand_craft,
            "readability": readability,
            "scaled_score": label,
            "encode_prompt": encode_prompt
        }
