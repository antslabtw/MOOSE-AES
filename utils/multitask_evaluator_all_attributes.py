from .metrics import *
from utils.general_utils import separate_attributes_for_scoring, separate_and_rescale_attributes_for_scoring


class Evaluator():
    def __init__(self, dev_dataset, test_dataset, seed):
        self.dev_dataset = list(dev_dataset)
        self.test_dataset = list(test_dataset)
        X_dev_prompt_ids = [sample["prompt_id"] for sample in self.dev_dataset]
        X_test_prompt_ids = [sample["prompt_id"] for sample in self.test_dataset]
        Y_dev = [sample["scaled_score"] for sample in self.dev_dataset]
        Y_test = [sample["scaled_score"] for sample in self.test_dataset]
        
        self.X_dev_prompt_ids, self.X_test_prompt_ids = X_dev_prompt_ids, X_test_prompt_ids
        self.Y_dev, self.Y_test = Y_dev, Y_test
        self.Y_dev_org = separate_and_rescale_attributes_for_scoring(self.Y_dev, self.X_dev_prompt_ids)
        self.Y_test_org = separate_and_rescale_attributes_for_scoring(self.Y_test, self.X_test_prompt_ids)
        self.best_dev_kappa_mean = -1
        self.best_test_kappa_mean = -1
        self.best_dev_kappa_set = {}
        self.best_test_kappa_set = {}
        self.seed = seed

    @staticmethod
    def calc_pearson(pred, original):
        pr = pearson(pred, original)
        return pr

    @staticmethod
    def calc_spearman(pred, original):
        spr = spearman(pred, original)
        return spr

    @staticmethod
    def calc_kappa(pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    @staticmethod
    def calc_rmse(pred, original):
        rmse = root_mean_square_error(original, pred)
        return rmse

    def evaluate(self, dev_pred, test_pred, epoch, print_info=True):
        self.current_epoch = epoch       
        dev_pred_dict = separate_and_rescale_attributes_for_scoring(dev_pred, self.X_dev_prompt_ids)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred, self.X_test_prompt_ids)

        for key in test_pred_dict.keys():
            print(f"Processing key: {key}")
            print(f"Pred shape: {len(test_pred_dict[key])}, Original shape: {len(self.Y_test_org[key])}")
            
        pearson_dev = {key: self.calc_pearson(dev_pred_dict[key], self.Y_dev_org[key]) for key in
                       dev_pred_dict.keys()}
        pearson_test = {key: self.calc_pearson(test_pred_dict[key], self.Y_test_org[key]) for key in
                        test_pred_dict.keys()}

        spearman_dev = {key: self.calc_spearman(dev_pred_dict[key], self.Y_dev_org[key]) for key in
                        dev_pred_dict.keys()}
        spearman_test = {key: self.calc_spearman(test_pred_dict[key], self.Y_test_org[key]) for key in
                        test_pred_dict.keys()}

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], self.Y_dev_org[key]) for key in
                        dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], self.Y_test_org[key]) for key in
                         test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch

        if print_info:
            self.print_info()

        results = {"dev_avg": self.dev_kappa_mean, "test_avg": self.test_kappa_mean}
        for att in self.kappa_dev.keys():
            results[f"dev_{att}"] = round(self.kappa_dev[att], 3)
            
        for att in self.kappa_test.keys():
            results[f"test_{att}"] = round(self.kappa_test[att], 3)

        return results

    def print_info(self):
        print('CURRENT EPOCH: {}'.format(self.current_epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(att, round(self.kappa_dev[att], 3)))
        print("-" * 24)
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(att, round(self.kappa_test[att], 3)))
        print("-" * 24)
        print('[BEST TEST] AVG QWK: {}, epoch: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print("-" * 122)

    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, epoch: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print("-" * 122)
        
