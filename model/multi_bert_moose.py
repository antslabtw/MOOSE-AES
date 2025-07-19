#document_bert_architectures
import torch
from torch import nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertConfig, BertModel
import torch.nn.functional as F
from data.scale import get_scaled_down_scores, separate_and_rescale_attributes_for_scoring
from utils.evaluate import evaluation
from utils.masks import get_trait_mask_by_prompt_ids
from tqdm import tqdm
from model.output import BertModelOutput


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h):
        w = torch.tanh(self.w(h))
        weight = self.v(w)
        weight = weight.squeeze(-1)
        weight = torch.softmax(weight, 1)
        weight = weight.unsqueeze(-1)

        out = torch.mul(h, weight.repeat(1, 1, h.size(2)))
        out = torch.sum(out, 1)
        return out


class DocumentBertSentenceChunkAttentionLSTM(BertPreTrainedModel):
    def __init__(self, args):
        bert_model_config = BertConfig.from_pretrained('bert-base-uncased')
        super(DocumentBertSentenceChunkAttentionLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        for layer in self.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False
        # for param in self.bert.parameters():#212121
        #     param.requires_grad = False
        
        self.dropout = nn.Dropout(0.5)
        self.lstm = LSTM(bert_model_config.hidden_size, bert_model_config.hidden_size, batch_first = True)
        self.w_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, bert_model_config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, bert_model_config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.linear = nn.Linear(bert_model_config.hidden_size, args.hidden_dim)

    def forward(self, document_batch: torch.Tensor, readability, hand_craft, device, bert_batch_size=0, length=0):
        bert_output = torch.zeros(size=(document_batch.shape[0],
                      document_batch.shape[1],
                      self.bert.config.hidden_size), dtype=torch.float, device=device)
        
        for doc_id in range(document_batch.shape[0]):
            bert_output_temp = self.bert(document_batch[doc_id][:length[doc_id], 0],
            token_type_ids = document_batch[doc_id][:length[doc_id], 1],
            attention_mask = document_batch[doc_id][:length[doc_id], 2])[1]

            bert_output_temp = torch.nan_to_num(bert_output_temp, nan=0.0)
            bert_output[doc_id][:length[doc_id]] = self.dropout(bert_output_temp)
        
        
        packed_input = pack_padded_sequence(bert_output, length, batch_first=True, enforce_sorted=False)
        packed_output, (_, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)


        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)  # (batch_size, seq_len, 1)
        attention_score = F.softmax(attention_u, dim=1)  # (batch_size, seq_len, 1)
        attention_hidden = output * attention_score  # (batch_size, seq_len, num_hiddens)
        attention_hidden = torch.sum(attention_hidden, dim=1)  # (batch_size, num_hiddens)
        attention_hidden = self.linear(attention_hidden)
        
        return attention_hidden


class DocumentBertCombineWordDocumentLinear(BertPreTrainedModel):
    def __init__(self, args):
        bert_model_config = BertConfig.from_pretrained('bert-base-uncased')
        super(DocumentBertCombineWordDocumentLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        for layer in self.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    
        self.bert_batch_size = 1
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(bert_model_config.hidden_size * 2, args.hidden_dim)
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, readability, hand_craft, device):

        # bert_output = torch.zeros(size=(document_batch.shape[0], 
        #               min(document_batch.shape[1], self.bert_batch_size), #分段的長度
        #               self.bert.config.hidden_size * 2),
        #               dtype=torch.float, device="cuda")
        # for doc_id in range(document_batch.shape[0]):
        #     # 只取分段一
        #     all_bert_output_info = self.bert(document_batch[doc_id][:self.bert_batch_size, 0],
        #                       token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
        #                       attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])
        #     bert_token_max = torch.max(all_bert_output_info[0], 1)
        #     bert_output[doc_id][:self.bert_batch_size] = torch.cat((bert_token_max.values, all_bert_output_info[1]), 1)
        # prediction = self.mlp(bert_output.view(bert_output.shape[0], -1))
        
        all_bert_output_info = self.bert(
                document_batch[:, 0, 0],
                token_type_ids = document_batch[:, 0, 1],
                attention_mask = document_batch[:, 0, 2]
        )
        bert_token_max = torch.max(all_bert_output_info[0], 1)
        bert_output = torch.cat((all_bert_output_info[1], bert_token_max.values), 1)
        #mlp
        prediction = self.mlp(bert_output)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class MultiTrait(nn.Module):
    def __init__(self, args):
        super(MultiTrait, self).__init__()
        dim = args.hidden_dim + 86 # 86 -> hand_craft + readability
        
        self.cross_atten = nn.MultiheadAttention(args.hidden_dim, args.mhd_head, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, args.hidden_dim),  
            nn.ReLU(),
            nn.Dropout(p=0.1),  
        )
        self.lnr = nn.Sequential(
            #nn.Linear(34, 100),  
            nn.LayerNorm(34),  
        )
        self.lnh = nn.Sequential(
            #nn.Linear(52, 52),  
            nn.LayerNorm(52),  
        )

        
    def forward(self, doc_fea, chunk_fea, readability, hand_craft, pdoc_fea, pchunk_fea, preadability, phand_craft):
        #readability = self.lnr(readability).unsqueeze(1)
        cross_out, _ = self.cross_atten(doc_fea.unsqueeze(1), chunk_fea, chunk_fea)
        cross_fea = cross_out.squeeze(1)
        #readability = self.lnr(readability)
        #hand_craft = self.lnh(hand_craft)
        #print(cross_fea.shape)
        #print(readability.shape)
        #print(hand_craft.shape)
        #print(cross_fea.max())
        #print(readability.max())
        #print(hand_craft.max())
        trait_fea = self.mlp(torch.cat([cross_fea, hand_craft, readability], dim=-1))
        #trait_fea = self.mlp(cross_fea)

        return trait_fea

# global pid

class Scorer(nn.Module):
    def __init__(self, args):
        super(Scorer, self).__init__()
        dim = args.hidden_dim  # 86 -> hand_craft + readability
        self.trait_att = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.trait_att2 = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.trait_att3 = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.score_layer = nn.Linear(dim, 1, bias=True)
        self.score_layer2 = nn.Linear(dim, 1, bias=True)
        self.score_layer3 = nn.Linear(dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()
        # self.c = 0
        
    def forward(self, x, mask, pdoc_fea, pchunk_fea):
        atten_out, _ = self.trait_att(x.detach(), x, x)
        atten_out = atten_out * mask.unsqueeze(-1)
        out = self.score_layer(atten_out).squeeze(-1)
        out = self.sigmoid(out)
        gat, atten_w = self.trait_att3(atten_out.detach(), torch.cat([pdoc_fea.unsqueeze(1), pchunk_fea], dim=1), torch.cat([pdoc_fea.detach().unsqueeze(1), pchunk_fea.detach()], dim=1))
        gat = gat * mask.unsqueeze(-1)
        gating = self.sigmoid3(self.score_layer3(gat))
        atten_out2, _ = self.trait_att2(atten_out.detach(), torch.cat([atten_out*gating+x*(1-gating), pdoc_fea.unsqueeze(1), pchunk_fea], dim=1), torch.cat([atten_out*gating+x*(1-gating), pdoc_fea.unsqueeze(1), pchunk_fea], dim=1))
        atten_out2 = atten_out2 * mask.unsqueeze(-1)
        out2 = self.score_layer2(atten_out2).squeeze(-1)
        out2 = self.sigmoid2(out2)

        # global pid
        # torch.save(atten_w.cpu(), f"record/{pid}/atten_w.{self.c}.pt")
        # self.c += 1
        return out, out2


class TraitSimilarity(nn.Module):

    def __init__(self, delta=0.7):
        super(TraitSimilarity, self).__init__()
        self.delta = delta
        
    def forward(self, y_pred, y_true):
        total_loss = torch.tensor(0.0, requires_grad=True, device=y_true.device)
        c = 0
        for j in range(1, y_true.size(1)):
            for k in range(j+1, y_true.size(1)):
                pcc = self.pearson_correlation_coefficient(y_true[:, j], y_true[:, k])
                cos = torch.cosine_similarity(y_pred[:, j], y_pred[:, k], dim=0)
                if pcc >= self.delta:
                    total_loss = total_loss + (1 - cos)
                c += 1
                
        return total_loss / c

    def pearson_correlation_coefficient(self, x, y):
        # Calculate the means of x and y
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        
        # Calculate the differences from the mean
        diff_x = x - mean_x
        diff_y = y - mean_y
        
        # Calculate the numerator and denominator for the correlation coefficient formula
        numerator = torch.sum(diff_x * diff_y)
        denominator_x = torch.sqrt(torch.sum(diff_x**2))
        denominator_y = torch.sqrt(torch.sum(diff_y**2))
        
        # Calculate the correlation coefficient
        r = numerator / (denominator_x * denominator_y)
        
        return r


class PairwiseRankLoss(nn.Module):

    def __init__(self):
        super(PairwiseRankLoss, self).__init__()

    def forward(self, logits, scores):
        batch_size = logits.size(0)
        total_mask = torch.ones((batch_size, batch_size))
        idx_pairs = torch.nonzero(total_mask).to(logits.device)
        
        logits_a, logits_b = logits[idx_pairs[:, 0]], logits[idx_pairs[:, 1]]
        scores_a, scores_b = scores[idx_pairs[:, 0]], scores[idx_pairs[:, 1]]
  
        term = (logits_a-logits_b)[scores_a>scores_b]
        total_loss = torch.clip(-term, min=0) + torch.log(1 + torch.exp(-torch.abs(term)))
        loss = total_loss.mean()
        return loss


class multiBert(nn.Module):

    def __init__(self, args):
        super(multiBert, self).__init__()
        self.chunk = DocumentBertSentenceChunkAttentionLSTM(args)
        self.linear = DocumentBertCombineWordDocumentLinear(args)
        self.pchunk = DocumentBertSentenceChunkAttentionLSTM(args)
        self.plinear = DocumentBertCombineWordDocumentLinear(args)
        self.multi_trait = nn.ModuleList([
            MultiTrait(args)
            for _ in range(args.num_trait)
        ])
        self.scorer = Scorer(args=args)
        self.hidden_dim = args.hidden_dim
        self.chunk_sizes = args.chunk_sizes
        self.mse_loss = nn.MSELoss()
        self.pairwise_rank_loss = PairwiseRankLoss()
        self.pooling = SoftAttention(args.hidden_dim)
        self.ts_loss = TraitSimilarity(args.delta)
        self.args = args
    
    def forward(self, prompt_id, document_single, chunked_documents, readability, hand_craft, scaled_score, prompt_document_single, prompt_chunked_documents, prompt_lengths, prompt_hand_craft, prompt_readability, lengths=0):
        # global pid
        # pid = prompt_id[0].cpu().item()
        
        #print(encode_prompt.shape)
        #print(hand_craft.shape)
        #print(readability.shape)
        #print(document_single.shape)
        #print(chunked_documents.shape)
        #print(prompt_hand_craft.shape)
        #print(prompt_readability.shape)
        #print(prompt_document_single.shape)
        
        device = self.args.device
        
        # single bert features
        prediction_single = self.linear(document_single, device=device, readability=readability, hand_craft=hand_craft) #(batch_size, 1)
        prompt_single = self.plinear(prompt_document_single, device=device, readability=prompt_readability, hand_craft=prompt_hand_craft) #(batch_size, 1)
        
        # chunked bert features
        prediction_chunked = torch.empty(prediction_single.shape[0], 0, self.hidden_dim, device=device)        
        for chunk_index in range(len(self.chunk_sizes)):
            batch_document_tensor_chunk = chunked_documents[chunk_index].to(device)
            length = lengths[chunk_index]
            length = length.cpu()
            predictions_chunk = self.chunk(batch_document_tensor_chunk, device=device, length=length, readability=readability, hand_craft=hand_craft)
            predictions_chunk = predictions_chunk.unsqueeze(1)
            prediction_chunked = torch.cat((prediction_chunked, predictions_chunk), dim=1) 
        prompt_chunked = torch.empty(prompt_single.shape[0], 0, self.hidden_dim, device=device)  
        for chunk_index in range(len(self.chunk_sizes)):
            batch_document_tensor_chunk = prompt_chunked_documents[chunk_index].to(device)
            #print(batch_document_tensor_chunk.shape)
            length = prompt_lengths[chunk_index]
            length = length.cpu()
            prompt_chunk = self.pchunk(batch_document_tensor_chunk, device=device, length=length, readability=prompt_readability, hand_craft=prompt_hand_craft)
            prompt_chunk = prompt_chunk.unsqueeze(1)
            prompt_chunked = torch.cat((prompt_chunked, prompt_chunk), dim=1)          
        
        # trait features
        trait_feas = torch.tensor([], requires_grad=True).to(device)
        for trait in self.multi_trait:
            trait_fea = trait(prediction_single, prediction_chunked, readability, hand_craft, prompt_single, prompt_chunked, prompt_readability, prompt_hand_craft)
            trait_feas = torch.cat([trait_feas, trait_fea.unsqueeze(1)], dim=1)
            
        scaled_score = scaled_score.to(device)
        mask = get_trait_mask_by_prompt_ids(prompt_id).to(device)
        pred_scores, pred_scores2 = self.scorer(trait_feas, mask, prompt_single, prompt_chunked)
        mask = self._get_mask(scaled_score)
        pair_mask = ~mask
        pr_loss = self.pairwise_rank_loss(pred_scores.masked_fill(pair_mask, -0.), scaled_score.masked_fill(pair_mask, -0.))
        pr_loss2 = self.pairwise_rank_loss(pred_scores2.masked_fill(pair_mask, -0.), scaled_score.masked_fill(pair_mask, -0.))
        
        mse_loss = self.mse_loss(pred_scores[mask], scaled_score[mask])
        mse_loss2 = self.mse_loss(pred_scores2[mask], scaled_score[mask])
        ts_loss = self.ts_loss(
            pred_scores.masked_fill(pair_mask, -0.), 
            scaled_score.masked_fill(pair_mask, -0.)
        )
        ts_loss2 = self.ts_loss(
            pred_scores2.masked_fill(pair_mask, -0.), 
            scaled_score.masked_fill(pair_mask, -0.)
        )
        beta = 0.2
        #loss = 0.7 * mse_loss  + 0.3 * ts_loss
        #loss = 0.5 * mse_loss  + 0.2 * ts_loss + 0.3 * pr_loss +  0.5 * mse_loss2  + 0.2 * ts_loss2 + 0.3 * pr_loss2
        loss = 0.7 * mse_loss  + 0.3 * ts_loss +  0.5 * mse_loss2  + 0.2 * ts_loss2 + 0.3 * pr_loss2
        
        return BertModelOutput(
            loss = loss,
            logits = pred_scores2,
            scores = scaled_score
        )

    def _get_mask(self, target):
        mask = torch.ones(*target.size(), device=target.device)
        mask.data.masked_fill_((target == -1), 0)
        return mask.to(torch.bool)
