import numpy as np

def get_score_vector_positions():
    return {'score': 0}


def get_min_max_scores():
    return {
        1: {'score': (2, 12)},
        2: {'score': (1, 6)},
        3: {'score': (0, 3)},
        4: {'score': (0, 3)},
        5: {'score': (0, 4)},
        6: {'score': (0, 4)},
        7: {'score': (0, 30)},
        8: {'score': (0, 60)}
    }



def get_scaled_down_scores(scores, prompts):
    score_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    scaled_score_list = []
    for score_vector, prompt in zip(scores, prompts):
        attribute_name = 'score' 
        min_val, max_val = min_max_scores[prompt][attribute_name]
        att_val = score_vector
        scaled_score = (att_val - min_val) / (max_val - min_val)
        scaled_score_list.append(scaled_score)  
    return scaled_score_list



def separate_and_rescale_attributes_for_scoring(scores, set_ids):
    min_max_scores = get_min_max_scores()
    rescaled_scores = []
    for score_vector, set_id in zip(scores, set_ids):
        min_score, max_score = min_max_scores[set_id]['score']
        rescaled_score = score_vector * (max_score - min_score) + min_score
        rescaled_scores.append(np.around(rescaled_score).astype(int))
    return rescaled_scores

