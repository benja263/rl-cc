import torch 
import numpy as np

from catboost import CatBoostRegressor

from neural_nets.mlp import MLP


MODEL_PATH = 'trained_models/rlcc_nn_model'

STATE_DIM = 2
ACTION_DIM = 1
HISTORY_LENGTH = 2
HIDDEN_SIZES = [12, 12]
TARGET = 0.768

FEATURE_WEIGHTS = [1, 1, 0.05 , 1]

TRAIN_VAL_RATIO = 0.8

def get_train_data():
    """
    Generate synthetic training data 
    """
    num_action_samples = 1000
    num_obs_samples = 5000
    data_set_size = int(10e6)

    action_space = np.linspace(0.8, 1.2, num_action_samples)
    
    observation_space = np.concatenate([np.linspace(-TARGET, TARGET, num_obs_samples), np.linspace(TARGET, 100*TARGET, num_obs_samples)])

    data = np.concatenate([np.random.choice(action_space, size=(data_set_size, 2)), np.random.choice(observation_space, size=(data_set_size, 2))], axis=1)
    data[:, [1, 2]] = data[:, [2, 1]]

    return data

def get_eval_data():
    """
    Generate evaluation data similar to TABLE VI in the paper
    """
    under_utilized = np.array([1.1, -TARGET])
    steady_state = np.array([1.0, 0])
    congested = np.array([1.1, TARGET])


    under_utilized = {'name': 'under-utilized', 'data': under_utilized}
    on_target = {'name': 'on target', 'data': steady_state}
    congested = {'name': 'congested', 'data': congested}
    return [under_utilized, on_target, congested]

def nn_predict(nn_model, states):
    with torch.no_grad():
        actions, _ = nn_model(torch.tensor(states).float())
        # map actions from [-1, 1] to [0.8, 1.2]
        actions[actions >= 0] = 1 + 0.2*actions[actions >= 0]
        actions[actions < 0] = 1 / (1 - 0.2*actions[actions < 0])
    return actions.cpu().numpy()

if __name__ == '__main__':
     # load nn model
    nn_model = MLP(input_size=STATE_DIM*HISTORY_LENGTH, output_size=ACTION_DIM, hidden_sizes=HIDDEN_SIZES,
                   activation='relu', use_rnn=None, bias=False)
    checkpoint_state_dict = torch.load(MODEL_PATH)
    nn_model.load_state_dict(checkpoint_state_dict['model_state_dict'])
    # get training data
    X = get_train_data()
    y = nn_predict(nn_model, X)
    split_idx = int(TRAIN_VAL_RATIO*len(y))
    X_tr, y_tr, X_test, y_test = X[:split_idx], y[:split_idx].flatten(), X[split_idx:], y[split_idx:].flatten()
    # configure catboost
    cb_params = {'loss_function': 'RMSE',
                'early_stopping_rounds': 50,
                'verbose': 1,
                'max_depth': 4,
                'learning_rate': 1,
                'iterations': 20,
                'l2_leaf_reg': 1,
                'min_data_in_leaf': 1,
                'task_type': 'CPU',
                'devices': "0:1",
                'feature_weights': FEATURE_WEIGHTS}
    # train distilled model 
    cbr = CatBoostRegressor(**cb_params)
    cbr.fit(X_tr, y_tr)

    print(f"Final model with {cbr.tree_count_} trees")
    y_pred = cbr.predict(X_test)
    mse = ((y_pred - y_test)**2).mean()
    print(f"MSE: {mse}")
   
    # get eval data 
    eval_states = get_eval_data()
    for prev_state in eval_states:
        for crnt_state in eval_states:
            print(f"current state: {crnt_state['name']} previous state: {prev_state['name']}")
            state = np.concatenate([crnt_state['data'], prev_state['data']], axis=-1)
            cb_pred = cbr.predict(state)
            nn_pred = nn_predict(nn_model, state)
            print(f"distilled action: {cb_pred} neural network actions: {nn_pred[0]}")






