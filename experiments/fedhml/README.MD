
## Quick Start
```
# general test 
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --batch_size 64 --client_optimizer adam
# --adjustment_epochs None --growth_data_mode batch --client_optimizer adam

# quick test 
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 1 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 
# --adjustment_epochs None --growth_data_mode batch --client_optimizer sgd
```

### NLP task
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedhml_distributed_pytorch.sh gpt2 tinystories 100 10 200 1 0.1 0.1 --delta_T 10 --T_end 100 --num_eval 128 --frequency_of_the_test 10 --partition_alpha 5 --batch_size 16

CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedhml_distributed_pytorch.sh gpt2 tinystories 100 10 500 5 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --partition_alpha 5 --batch_size 16
```



## Usage
[$\cdot$] is mandatory arguments, {$\cdot$} is the optional arguments.
```
CUDA_VISIBLE_DEVICES=[gpus] sh run_fedhml_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round]  [comm_round] [epochs] [target_density] [initial_lr]   {--delta_T , --T_end, --partition_alpha, --num_eval, --frequency_of_the_test}
```

where

```
[gpus] specifies which GPUs to use.
[model] is the name of the model.
[dataset] is the name of the dataset.
[client_num_in_total] is the total number of clients.
[client_num_per_round] is the number of clients selected per round.
[comm_round] is the number of communication rounds.
[epochs] is the number of local epochs.
[target_density] is the target density for the sparse model.
[initial_lr] is the initial learning rate.
{--delta_T} is the interval rounds between two adjustment round.
{--T_end} is the end round number for adjustment round.
{--partition_alpha} refers to the partition alpha, higher partition alpha makes lower degree of data heterogeneity
{--num_eval}is the number data samples for validation, -1 means using the whole testing dataset.
{--frequency_of_the_test} the frequency to test/validate the performance during the training, using num_eval data samples
```

## Experiment

cd /data1/jxye/FedPruning_up/experiments/fedhml && conda activate fedpruning

fuser -v /dev/nvidia5 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
-------resnet18---------

****cifar 10
CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10


CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****cinic 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh resnet18 cinic10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh resnet18 cinic10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh resnet18 cinic10 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh resnet18 cinic10 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****svhn

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 svhn 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 svhn 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 svhn 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 svhn 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****tinyimagenet

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 tinyimagenet 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 tinyimagenet 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 tinyimagenet 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 tinyimagenet 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

-------shufflenet v2---------

****cifar 10
CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh shufflenet cifar10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh shufflenet cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh shufflenet cifar10 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh shufflenet cifar10 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****cinic 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh shufflenet cinic10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh shufflenet cinic10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh shufflenet cinic10 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh shufflenet cinic10 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****svhn

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh shufflenet svhn 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh shufflenet svhn 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh shufflenet svhn 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh shufflenet svhn 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****tinyimagenet

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh shufflenet tinyimagenet 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh shufflenet tinyimagenet 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh shufflenet tinyimagenet 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh shufflenet tinyimagenet 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

-------vit---------

****cifar 10
CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh vit cifar10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh vit cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh vit cifar10 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh vit cifar10 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****cinic 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh vit cinic10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh vit cinic10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh vit cinic10 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4  sh run_fedhml_distributed_pytorch.sh vit cinic10 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****svhn

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh vit svhn 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh vit svhn 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh vit svhn 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh vit svhn 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

****tinyimagenet

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh vit tinyimagenet 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh vit tinyimagenet 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh vit tinyimagenet 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh vit tinyimagenet 100 10 500 5 0.5 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

*********tinystories gpt2

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 sh run_fedhml_distributed_pytorch.sh gpt2 tinystories 100 10 150 1 0.2 0.2 --delta_T 10 --T_end 300 --partition_alpha 10 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh gpt2 tinystories 100 10 150 1 0.3 0.2 --delta_T 10 --T_end 300 --partition_alpha 10 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh gpt2 tinystories 100 10 150 1 0.5 0.2 --delta_T 10 --T_end 300 --partition_alpha 10 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh gpt2 tinystories 100 10 150 1 0.2 0.1 --gpu_mapping_file gpu_mapping.yaml --gpu_mapping_key mapping_default --delta_T 10 --T_end 100 --partition_alpha 5 --growth_data_mode batch --client_optimizer sgd --num_eval 128 --frequency_of_the_test 10 --batch_size 16

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh gpt2 tinystories 100 10 150 1 0.3 0.1 --gpu_mapping_file gpu_mapping.yaml --gpu_mapping_key mapping_default --delta_T 10 --T_end 100 --partition_alpha 5 --growth_data_mode batch --client_optimizer sgd --num_eval 128 --frequency_of_the_test 10 --batch_size 16

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh gpt2 tinystories 100 10 150 1 0.5 0.1 --gpu_mapping_file gpu_mapping.yaml --gpu_mapping_key mapping_default --delta_T 10 --T_end 100 --partition_alpha 5 --growth_data_mode batch --client_optimizer sgd --num_eval 128 --frequency_of_the_test 10 --batch_size 16

----------------Data Heterogeneity--------------

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.1 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.3 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 1 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 99999 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

----------------Different client num--------------

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 10 1 500 5 0.2 1 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer sgd --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 10 3 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 10 5 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 10 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10

------------------------Ablation study
*target density

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10 --lambda_etr 0

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10 --lambda_etr 0

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10 --lambda_etr 0

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10 --lambda_etr 1

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.2 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10 --lambda_etr 1

CUDA_VISIBLE_DEVICES=1,2,3,4 sh run_fedhml_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.3 0.001 --delta_T 10 --T_end 300 --partition_alpha 0.5 --growth_data_mode batch --client_optimizer adam --num_eval 128 --frequency_of_the_test 10 --lambda_etr 1
