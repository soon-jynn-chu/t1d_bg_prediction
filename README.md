# Blood Glucose Prediction in T1D using Supervised ML

## Set Up
- Python version: 3.10.14
    ```bash
    pip install -r requirements.txt
    ```

## Example usage
- Traditional ML
    ```python
    python train_ml.py --data_path /path/to/folder --exp_name svr
    ```
- DNN
    ```python
    python train_dnn.py --data_path /path/to/folder --exp_name gru --hidden_size 86 --num_layers 2
    ```

- DRL
    ```python
    python train_drl.py --data_path /path/to/folder --exp_name ddpg --epochs 8 --action_size 6
    ```

- Voting
    ```python
    # Require prediction outputs from base estimators
    python train_vote.py --output_path /path/to/output/folder --model_names svr rf lgb
    ```

- Stacking
    ```python
    # Require prediction outputs from base estimators
    python train_vote.py --output_path /path/to/output/folder --model_names svr rf lgb
    ```