
class loss_holder:
    def __init__(self, tqdm_instance, len_set=100):
        self.value = 0.0
        self.iter = 0
        self.tqm_instance = tqdm_instance
        self.LOSS_UPDATE_FREQ = len_set // 100 + 1

    def update(self, loss):
        self.value += loss

        # print statistics every LOSS_UPDATE_FREQ mini batches
        if self.iter % self.LOSS_UPDATE_FREQ == 0:
            stats = {"loss": "{:.5f}".format(self.value/self.iter)}
            self.tqm_instance.set_postfix(stats)
        self.iter += 1