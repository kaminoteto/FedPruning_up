from .utils import transform_tensor_to_list
from .fedhml_utils import compute_topological_mask


class FedHMLTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self, mode, round_idx=None):
        gradients = self.trainer.train(self.train_local, self.device, self.args, mode, round_idx)
        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)

        # The topological mask is computed on the server side, not returned here.
        return weights, gradients, self.local_sample_number

    def test(self, test_data, device, args):
        # Wrapper for the underlying model_trainer test if needed
        return self.trainer.test(test_data, device, args)