# Homework 2 in HKU-DATA8003: Edge of Stability (EOS) Lab 
This is the code workspace of the homework 2 in DATA8003. 


## Install
```
pip install -r requirements.txt
```

## Run
```
cd codes
python main.py +experiments/<TASK>=<CONFIG_NAME>
```

## Visualization
Once you have run the code, you can find the visualization results in the `experiments/<EXPERIMENT_NAME>/hist` folder. Moreover, if you have configured the `wandb` in the `logger/wandb.yaml` file, you can also find the visualization results in the [wandb](https://wandb.ai/).


## Acknowledgement
Thanks for the [EOS](https://github.com/locuslab/edge-of-stability) for providing the key code of the framework.