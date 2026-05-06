import torch.nn as nn

class GLU(nn.Module):
    '''
    Gating Linear Unit.
    Inputs: input, context, output_size
    '''
    def __init__(self, input_size=384, context_size=1024, output_size=1024):
        super().__init__()
        self.fc_1 = nn.Linear(context_size, input_size)
        self.fc_2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context):
        # context shape: [batch_size x context_size]
        gate = self.sigmoid(self.fc_1(context))
        # gate shape: [batch_size x input_size]

        # The line is the same as below: gated_input = torch.mul(gate, x)
        # x shape: [batch_size x input_size]
        gated_input = x * (1 + gate)

        # gated_input shape: [batch_size x input_size]
        output = self.fc_2(gated_input)

        # del context, x, gate, gated_input

        return output