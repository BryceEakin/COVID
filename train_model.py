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

    config.optim_adam_betas = (0.992, 0.9995)

    if os.name == 'nt' and False:
        run_thread = threading.Thread(
            target=train_model, 
            args=[config], 
            kwargs={
                'check_interrupted':check_interrupted,
                #'disable_checkpointing':True,
                'debug':True
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
        train_model(config, debug=True)
