# Federated Optimization for Heterogeneous Networks



## Preparation

### Dataset generation

This repository **already includes four synthetic datasets** that are used in the paper. For all datasets, see the `README` files in separate `data/$dataset` folders for instructions on preprocessing and/or sampling data.



## Run on synthetic federated data 
(1) Make sure you have `pip3` in your machine and could successfully download those libraries:

```
pip3 install -r requirements.txt  
```

(2) You don't need a GPU to run the synthetic data experiments:

```
export CUDA_VISIBLE_DEVICES=
```

(3) Run `bash run.sh dataset_name value_of_mu` as follows, and the log files are automatically stored for drawing figures later.

*Each of the instructions may take about half an hour to run on CPUs, but you can terminate them at an earlier stage (e.g., at about 40 rounds) and draw figures as instructed in (4), to quickly check that the performance is the same as reported.*

```
bash run.sh synthetic_iid 0 | tee log_synthetic/synthetic_iid_client10_epoch20_mu0
bash run.sh synthetic_iid 1 | tee log_synthetic/synthetic_iid_client10_epoch20_mu1
bash run.sh synthetic_0_0 0 | tee log_synthetic/synthetic_0_0_client10_epoch20_mu0
bash run.sh synthetic_0_0 1 | tee log_synthetic/synthetic_0_0_client10_epoch20_mu1
bash run.sh synthetic_0.5_0.5 0 | tee log_synthetic/synthetic_0.5_0.5_client10_epoch20_mu0
bash run.sh synthetic_0.5_0.5 1 | tee log_synthetic/synthetic_0.5_0.5_client10_epoch20_mu1
bash run.sh synthetic_1_1 0 | tee log_synthetic/synthetic_1_1_client10_epoch20_mu0
bash run.sh synthetic_1_1 1 | tee log_synthetic/synthetic_1_1_client10_epoch20_mu1
```

(4) Draw figures to reproduce results on synthetic data (Figure 1 and Figure 6 in the paper)

```
python plot.py loss     # training loss
python plot.py accuracy # testing accuracy
python plot.py dissim   # dissimilarity metric

```


The training loss, testing accuracy and dissimilarity measurement figures are saved as `loss.pdf`, `accuracy.pdf` and `dissim.pdf` respectively, under the current folder where you call `plot.py`. Make sure to use the default hyper-parameters in `run.sh` for synthetic data.

## Run on real federated datasets

(1) First, make sure to install all packages:

```
pip3 install -r requirements.txt
```
(2) Then specify a GPU id if you have want to use GPUs:

```
export CUDA_VISIBLE_DEVICES=available_gpu_id
```
Otherwise just run to CPUs:

```
export CUDA_VISIBLE_DEVICES=
```

(3) Run the following instruction, replace `$DATASET` with a dataset of interest (see all dataset names in the `FedProx/data` directory), specify the corresponding model to that dataset (choose from `flearn/models/$DATASET/$MODEL.py` and use `$MODEL` as the model name), specify a log file name, and configure all other parameters (we report all the hyper-parameters in the appendix of the paper):

```
mkdir logs
python3  -u main.py --dataset=$DATASET --optimizer='fedprox'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --mu=0 --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model=$MODEL | tee logs/$LOG_FILE_NAME 
```

And the log file which contains accuracy, loss and dissimilarity numbers would be saved under `logs/`.

*Note: It might take a much longer time to run on real datasets than synthetic data because real federated datasets are larger and some of the models are deep neural networks.*


## References
See our [FedProx](https://arxiv.org/abs/1812.06127)  manuscript for more details as well as all references.