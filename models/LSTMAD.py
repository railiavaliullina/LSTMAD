import torch


class LSTM_AD(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.lstm1 = torch.nn.LSTMCell(self.cfg.d, self.cfg.hidden_size)
        self.lstm2 = torch.nn.LSTMCell(self.cfg.hidden_size, self.cfg.hidden_size)
        self.linear = torch.nn.Linear(self.cfg.hidden_size, self.cfg.d * self.cfg.out_dim)

        self.h_t = torch.autograd.Variable(torch.zeros(self.cfg.batch_size, self.cfg.hidden_size), requires_grad=False)
        self.c_t = torch.autograd.Variable(torch.zeros(self.cfg.batch_size, self.cfg.hidden_size), requires_grad=False)
        self.h_t2 = torch.autograd.Variable(torch.zeros(self.cfg.batch_size, self.cfg.hidden_size), requires_grad=False)
        self.c_t2 = torch.autograd.Variable(torch.zeros(self.cfg.batch_size, self.cfg.hidden_size), requires_grad=False)

    def forward(self, input_x):
        outputs = []
        h_t, c_t, h_t2, c_t2 = self.h_t, self.c_t, self.h_t2, self.c_t2

        for input_t in input_x.chunk(input_x.size(1), dim=1):
            h_t, c_t = self.lstm1(input_t.squeeze(dim=1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.extend(output)
        outputs = torch.stack(outputs, 1).squeeze()
        outputs = outputs.view(input_x.size(0), input_x.size(1), self.cfg.d, self.cfg.out_dim)
        return outputs
