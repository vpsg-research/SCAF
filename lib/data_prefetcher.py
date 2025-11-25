import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target, self.p_fg, self.p_bg, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.p_fg = self.p_fg.cuda(non_blocking=True)
            self.p_bg = self.p_bg.cuda(non_blocking=True)
            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need
            self.p_fg = self.p_fg.float()
            self.p_bg = self.p_bg.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        p_fg = self.p_fg
        p_bg = self.p_bg
        self.preload()
        # image, mask, p_fg, p_bg
        return input, target, p_fg, p_bg
