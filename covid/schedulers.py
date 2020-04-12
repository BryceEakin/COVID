__ALL__ = [
    'LinearWarmupScheduler'
]

class LinearWarmupScheduler(object):
    def __init__(self, optimizer, steps, last_step=-1):
        self.optimizer = optimizer
        self.last_step = last_step
        self.steps = steps
        
        self.max_lr = [x['lr'] for x in optimizer.param_groups]

        self.step()

    def state_dict(self):
        return { 'max_lr': self.max_lr, 'last_step': self.last_step }

    def load_state_dict(self, val):
        self.max_lr = val['max_lr']
        self.last_step = val['last_step']

    def step(self):
        self.last_step += 1
        if self.last_step >= self.steps:
            return

        current_lr_factor = (self.last_step + 1)/self.steps

        for group, lr in zip(self.optimizer.param_groups, self.max_lr):
            group['lr'] = lr * current_lr_factor