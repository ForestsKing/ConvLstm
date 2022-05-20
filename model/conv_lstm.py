import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, input2state_kernel_size, state2state_kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state2state_kernel_size = state2state_kernel_size
        self.input2state_kernel_size = input2state_kernel_size
        self.state2state_padding = state2state_kernel_size[0] // 2, state2state_kernel_size[1] // 2
        self.input2state_padding = input2state_kernel_size[0] // 2, input2state_kernel_size[1] // 2

        self.W_x = nn.Conv2d(in_channels=self.input_dim,
                             out_channels=self.hidden_dim * 4,
                             kernel_size=self.input2state_kernel_size,
                             padding=self.input2state_padding
                             )
        self.W_h = nn.Conv2d(in_channels=self.hidden_dim,
                             out_channels=self.hidden_dim * 4,
                             kernel_size=self.state2state_kernel_size,
                             padding=self.state2state_padding
                             )

    def forward(self, x, h_pre, c_pre):
        """
        :param x: batch_size * input_dim * weight * height
        :param h_pre: batch_size * hidden_dim * weight * height
        :param c_pre: batch_size * hidden_dim * weight * height
        :return: h_next: batch_size * hidden_dim * weight * height
        :return: c_next: batch_size * hidden_dim * weight * height
        """

        conv_xi, conv_xf, conv_xc, conv_xo = torch.split(self.W_x(x), self.hidden_dim, dim=1)
        conv_hi, conv_hf, conv_hc, conv_ho = torch.split(self.W_h(h_pre), self.hidden_dim, dim=1)

        i = torch.sigmoid(conv_xi + conv_hi)
        f = torch.sigmoid(conv_xf + conv_hf)
        c_next = f * c_pre + i * torch.tanh(conv_xc + conv_hc)
        o = torch.sigmoid(conv_xo + conv_ho)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, input2state_kernel_size, state2state_kernel_size, num_layers):
        super(ConvLSTMModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state2state_kernel_size = state2state_kernel_size
        self.input2state_kernel_size = input2state_kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            tmp_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(
                ConvLSTMCell(tmp_input_dim, self.hidden_dim, self.input2state_kernel_size, self.state2state_kernel_size)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, h_0=None, c_0=None):
        batch_size, seq_len, _, height, weight = x.size()

        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_dim, height, weight, self.num_layers).to(x.device)
        if c_0 is None:
            c_0 = torch.zeros(batch_size, self.hidden_dim, height, weight, self.num_layers).to(x.device)

        hidden, cell = [], []
        output = None
        for layer in range(self.num_layers):
            output = x if layer == 0 else output
            h, c = h_0[:, :, :, :, layer], c_0[:, :, :, :, layer]

            out = []
            for i in range(seq_len):
                h, c = self.cell_list[layer](output[:, i, :, :, :], h, c)
                out.append(h)

            output = torch.stack(out, dim=1)
            hidden.append(h)
            cell.append(c)

        hidden = torch.stack(hidden, dim=-1)
        cell = torch.stack(cell, dim=-1)
        return output, hidden, cell


class ConvLSTM(nn.Module):
    def __init__(self,
                 input_dim=1,
                 hidden_dim=64,
                 pred_len=10,
                 input2state_kernel_size=(5, 5),
                 state2state_kernel_size=(5, 5),
                 num_layers=2):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.input2state_kernel_size = input2state_kernel_size
        self.state2state_kernel_size = state2state_kernel_size
        self.num_layers = num_layers

        self.encoder = ConvLSTMModule(input_dim=self.input_dim,
                                      hidden_dim=self.hidden_dim,
                                      input2state_kernel_size=input2state_kernel_size,
                                      state2state_kernel_size=state2state_kernel_size,
                                      num_layers=num_layers
                                      )
        self.decoder = ConvLSTMModule(input_dim=self.input_dim,
                                      hidden_dim=self.hidden_dim,
                                      input2state_kernel_size=input2state_kernel_size,
                                      state2state_kernel_size=state2state_kernel_size,
                                      num_layers=num_layers
                                      )
        self.fc = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=(1, 1))

    def forward(self, x):
        batch_size, _, _, height, weight = x.size()

        _, hidden, cell = self.encoder(x)

        pred = []
        temp_input = torch.zeros((batch_size, 1, self.input_dim, height, weight), dtype=torch.float).to(x.device)
        for t in range(self.pred_len):
            temp_input, hidden, cell = self.decoder(temp_input, hidden, cell)
            temp_input = self.fc(torch.squeeze(temp_input, dim=1))
            pred.append(temp_input)
            temp_input = torch.unsqueeze(temp_input, dim=1)

        pred = torch.stack(pred, dim=1)
        return pred
