import logging
import os, signal
import sys

from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process
from .fedhml_utils import compute_topological_mask # Import the HML logic

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
try:
    from core.distributed.communication.message import Message
    from core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedPruning.core.distributed.communication.message import Message
    from FedPruning.core.distributed.server.server_manager import ServerManager

class FedHMLServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.mode = 0 

    def run(self):
        super().run()

    def send_init_msg(self):
        logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1], self.mode, self.round_idx)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                self.handle_message_receive_model_from_client)

    def mode_convert(self,):
        if self.mode == 0:
            if self.round_idx % self.args.delta_T == 0 and self.round_idx <= self.args.T_end :
                self.mode = 2
            else:
                self.mode = 1
        elif self.mode == 1:
            if self.round_idx % self.args.delta_T == 0 and self.round_idx <= self.args.T_end :
                self.mode = 2
            else:
                self.mode = 1
        elif self.mode == 2:
            self.mode = 0
        elif self.mode == 3:
            if self.round_idx < self.args.T_end :
                self.mode = 3
            else:
                self.mode = 0

        return self.mode
    
    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        
        if self.mode in [2, 3]:
            gradients = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADIENT)
            self.aggregator.add_local_trained_gradient(sender_id - 1, gradients)
            
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        
        if b_all_received:
            # 1. Aggregate Weights (Returns dict, NOT tuple)
            global_model_params = self.aggregator.aggregate()
            
            # 2. Logic for Mask Adjustment
            if self.mode in [2, 3]:
                global_gradient = self.aggregator.aggregate_gradient()
                
                # Check for None to prevent crashes
                if global_gradient is not None:
                    # In FedHML, we use Topological Pruning (Homology)
                    # We compute the mask on the updated Global Model
                    logging.info("Calculating Topological Mask via Homology...")
                    
                    # Compute mask based on the global model structure
                    new_topo_mask = compute_topological_mask(self.aggregator.trainer.model, self.args.target_density)
                    
                    # Apply this new mask to the server model
                    self.aggregator.trainer.model.mask_dict = new_topo_mask
                    self.aggregator.trainer.model.apply_mask()
                else:
                     logging.warning("Global gradient is None, skipping mask adjustment")

            # Testing
            self.aggregator.test_on_server_for_all_clients(self.round_idx)
            
            # Prepare for next round
            self.round_idx += 1
            self.mode = self.mode_convert()

            if self.round_idx == self.round_num + 1:
                self.finish()
                return

            # Client Sampling
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                    self.args.client_num_per_round)

            logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
            
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)
            
            # 3. Broadcast to clients (Send Mask if in Mode 0 or 3)
            if self.mode in [0, 3]:
                # Get the current mask from the server model
                mask_dict = self.aggregator.trainer.model.mask_dict
                for k in mask_dict:
                    mask_dict[k] = mask_dict[k].cpu()
                    
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx, mask_dict)
            else:
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx)


    def send_message_init_config(self, receive_id, global_model_params, client_index, mode_code, round_idx):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, mode_code, round_idx,
        mask_dict=None):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_MASKS, mask_dict)
        self.send_message(message)