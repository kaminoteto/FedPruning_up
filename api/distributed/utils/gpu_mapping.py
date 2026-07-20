import logging
import socket

import torch
import yaml


# def mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, gpu_util_file, gpu_util_key):
#     if gpu_util_file == None:
#         device = torch.device("cpu")
#         logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         logging.info(" ################## You do not indicate gpu_util_file, will use CPU training  #################")
#         logging.info(device)
#         # return gpu_util_map[process_id][1]
#         return device
#     else:
#         with open(gpu_util_file, 'r') as f:
#             gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
#             # gpu_util_num_process = 'gpu_util_' + str(worker_number)
#             # gpu_util = gpu_util_yaml[gpu_util_num_process]
#             gpu_util = gpu_util_yaml[gpu_util_key]
#             logging.info("gpu_util = {}".format(gpu_util))
#             gpu_util_map = {}
#             i = 0
#             for host, gpus_util_map_host in gpu_util.items():
#                 for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
#                     for _ in range(num_process_on_gpu):
#                         gpu_util_map[i] = (host, gpu_j)
#                         i += 1
#             logging.info("Process %d running on host: %s, gethostname: %s, local_gpu_id: %d ..." % (
#                 process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1]))
#             logging.info("i = {}, worker_number = {}".format(i, worker_number))
#             assert i == worker_number
#         if torch.cuda.is_available():
#             torch.cuda.set_device(gpu_util_map[process_id][1])
#         device = torch.device("cuda:" + str(gpu_util_map[process_id][1]) if torch.cuda.is_available() else "cpu")
#         logging.info("process_id = {}, GPU device = {}".format(process_id, device))
#         # return gpu_util_map[process_id][1]
#         return device

def mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, gpu_util_file, gpu_util_key):
    if gpu_util_file is None:
        # 原有CPU逻辑不变
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ################## You do not indicate gpu_util_file, will use CPU training  #################")
        logging.info(device)
        return device
    else:
        with open(gpu_util_file, 'r') as f:
            gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
            gpu_util = gpu_util_yaml[gpu_util_key]
            logging.info("gpu_util = {}".format(gpu_util))
            gpu_util_map = {}
            i = 0
            
            # 核心修改：动态分配进程到GPU，确保总数=worker_number
            # 步骤1：获取所有可用的GPU节点（host + gpu_id）
            available_gpus = []
            for host, gpus_util_map_host in gpu_util.items():
                for gpu_j in range(len(gpus_util_map_host)):
                    available_gpus.append((host, gpu_j))
            
            # 步骤2：将worker_number个进程均匀分配到可用GPU
            if not available_gpus:
                raise ValueError("No GPU configured in gpu_util_file!")
            for process_idx in range(worker_number):
                # 轮询分配进程到GPU
                gpu_idx = process_idx % len(available_gpus)
                host, gpu_j = available_gpus[gpu_idx]
                gpu_util_map[process_idx] = (host, gpu_j)
            i = worker_number  # 强制i等于worker_number
            
            logging.info("Process %d running on host: %s, gethostname: %s, local_gpu_id: %d ..." % (
                process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1]))
            logging.info("i = {}, worker_number = {}".format(i, worker_number))
            assert i == worker_number  # 此时断言必然通过
        
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_util_map[process_id][1])
        device = torch.device("cuda:" + str(gpu_util_map[process_id][1]) if torch.cuda.is_available() else "cpu")
        logging.info("process_id = {}, GPU device = {}".format(process_id, device))
        return device