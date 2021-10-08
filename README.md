# PRL-Project
Pratical Reinforcement Learning Course Project (Atari-Breakout)

## Table Of Contents:
* [Directory Structure](#directory-structure)
* [Steps to Run Tests](#steps-to-run-tests)
* [Training Procedure](#training-procedure)
* [Checkpoint Location](#checkpoint-location)


## Directory Structure:
```
.
|
|   |-- frames(the data)
|   |    |-- stack
|   |    |-- nostack
|   |-- logs
|   |    |-- training.log
|   |-- agent
|   |    |-- model1
|   |        |-- videos(all test videos)
|   |    |-- model2
|   |        |-- videos(all test videos)
|   |    |-- model3
|   |        |-- videos(all test videos)
|   |-- saved
|   |    |-- model1
|   |        |-- checkpoints(from training)
|   |    |-- model2
|   |        |-- checkpoints(from training)
|   |    |-- model3
|   |        |-- checkpoints(from training)
|    datasets.py
|    gendata_nostack.py
|    gendata_stack.py
|    models.py
|    README.md 
|    test_nostack.py
|    test_stack.py
|    train_nostack.py
|    train_stack.py
```

## Steps to Run Tests:
1. 
    * For Model 1 & Model 2:
        ```
        python test_stack.py --name=[model1|model2] --max_steps=N (max number of steps in one episode) --ckpt=ckpt_name (only name of the checkpoint not the path, without '.pt' extension)
        ```
        For example:
        ```
        python test_stack.py --name=model2 --max_steps=30000 --ckpt=ckpt_8
        ```
    * Model 3:
        ```
        python test_nostack.py --max_steps=N (max number of steps in one episode) --ckpt=ckpt_name (only name of the checkpoint not the path, without '.pt' extension)
        ```
        For example:
        ```
        python test_nostack.py --max_steps=30000 --ckpt=ckpt_5
        ```
2. The output of the above statement will be present in the agent folder under the corresponding model's folder.(The highest test_n will be the latest run)

## Training Procedure:
* Model 1 & Model 2:
    ```
    python gendata_stack.py
    ``` 
    this is to play the game and generate data. Also make sure 'frames/stack' is empty.
    ```
    python train_stack.py --name=[model1|model2] --ckpt=ckpt_name --epochs=N --batchs=M --learning_rate=float
    ```

* Model 3:
    ```
    python gendata_nostack.py
    ```
    this is to play the game and generate data. Also make sure 'frames/nostack' is empty.
    ```
    python train_nostack.py --ckpt=ckpt_name --epochs=N --batchs=M --learning_rate=float
    ```

```models.py``` contains all the required model definitions.<br/>
```datasets.py``` contains all the required pre-processing logic.

## Checkpoint Location:
The checkpoints are stored in the saved directory within the corresponding model folder.