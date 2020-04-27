from covid.training import train_model, CovidTrainingConfiguration
from covid.utils import getch

import threading
import os

import logging

if __name__ == '__main__':
    
    interrupted = False
    def check_interrupted():
        global interrupted
        return interrupted
    
    config = CovidTrainingConfiguration()
    config.verbosity = logging.DEBUG

    config.training_fold = 0
    config.optim_adam_betas = (0.98, 0.9995)
    config.chem_hidden_size = 512
    config.chem_layers_per_message = 3
    config.chem_nonlinearity = 'ReLU'
    config.dropout_rate = 0.1
    config.optim_adam_eps = 1e-4
    config.optim_initial_lr = 1e-4
    config.protein_base_dim = 16
    config.protein_downscale_nonlinearity = 'silu'
    config.protein_nonlinearity = 'silu'
    config.protein_output_dim = 256
    config.synthetic_negative_rate = 0.5

    if os.name == 'nt':
        run_thread = threading.Thread(
            target=train_model, 
            args=[config], 
            kwargs={
                'check_interrupted':check_interrupted,
                #'disable_checkpointing':True
            }
        )
        run_thread.start()

        print("Press 'q' or 'ctrl+c' to interrupt training loop")
            
        while True:
            ch = getch()
            if ch in (b'q', 'q', b'\x03', '\x03'):
                interrupted = True
                break
        
        print("Trying to quit....")
        run_thread.join()
    else:
        train_model(config)
