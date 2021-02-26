import torch
from eluvio.utils.constants import *


class BoundaryDifference(torch.nn.Module):
    def __init__(self):
        super(BoundaryDifference, self).__init__()
        self.conv2d = torch.nn.Conv2d(1, 512, kernel_size=(1,1))
        self.cosine = torch.nn.functional.cosine_similarity

    def forward(self, x):
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        bd_1, bd_2 = torch.split(x, [1]*2, dim=2)
        bd_1 = self.conv2d(bd_1)
        bd_1 = bd_1.view(bd_1.shape[0],bd_1.shape[1], -1)
        bd_2 = self.conv2d(bd_2)
        bd_2 = bd_2.view(bd_2.shape[0],bd_2.shape[1], -1)
        bd = self.cosine(bd_1, bd_2, dim=2)
        return bd

class BoundaryRelation(torch.nn.Module):
    def __init__(self):
        super(BoundaryRelation, self).__init__()
        self.conv2d = torch.nn.Conv2d(1, 512, kernel_size=(2, 1))
        self.max3d = torch.nn.MaxPool3d(kernel_size=(512,1,1))
    
    def forward(self, x):
        x = x.view(x.shape[0]*x.shape[1], 1, -1, x.shape[-1])
        x = self.conv2d(x)
        x = self.max3d(x)
        x = x.view(x.shape[0],-1)
        # x.squeeze()
        return x

class BoundaryNet(torch.nn.Module):
    def __init__(self):
        super(BoundaryNet, self).__init__()
        self.B_r = BoundaryRelation()
        self.B_d = BoundaryDifference()

    def forward(self, x):
        br = self.B_r(x)
        bd = self.B_d(x)
        return torch.cat((br, bd), dim=1)

class LGSceneSeg(torch.nn.Module):
    def __init__(self):
        super(LGSceneSeg, self).__init__()
        self.BNet_place = BoundaryNet()
        self.BiLstm_place = torch.nn.LSTM(
            input_size=PLACE_DIM+SIMI_DIM,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.BNet_cast = BoundaryNet()
        self.BiLstm_cast = torch.nn.LSTM(
            input_size=CAST_DIM+SIMI_DIM,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.BNet_action = BoundaryNet()
        self.BiLstm_action = torch.nn.LSTM(
            input_size=ACTION_DIM+SIMI_DIM,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.BNet_audio = BoundaryNet()
        self.BiLstm_audio = torch.nn.LSTM(
            input_size=AUDIO_DIM+SIMI_DIM,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.fc1_p = torch.nn.Linear(1024,100)
        self.fc2_p = torch.nn.Linear(100,2)
        self.fc1_c = torch.nn.Linear(1024,100)
        self.fc2_c = torch.nn.Linear(100,2)
        self.fc1_ac = torch.nn.Linear(1024,100)
        self.fc2_ac = torch.nn.Linear(100,2)
        self.fc1_au = torch.nn.Linear(1024,100)
        self.fc2_au = torch.nn.Linear(100,2)
    
    def forward(self, place, cast, action, audio):
        # Place
        bs=place.shape[0]
        p=place.shape[1]
        bnp = self.BNet_place(place)
        bnp = bnp.view(bs,p,bnp.shape[-1])
        self.BiLstm_place.flatten_parameters()
        out_p, (hp, cp) = self.BiLstm_place(bnp, None)
        out_p = torch.nn.functional.relu(self.fc1_p(out_p))
        out_p = self.fc2_p(out_p)
        out_p = out_p.view(-1,2)
        
        # Cast
        bnc = self.BNet_cast(cast)
        bnc = bnc.view(bs,p,bnc.shape[-1])
        self.BiLstm_cast.flatten_parameters()
        out_c, (hc, cc) = self.BiLstm_cast(bnc, None)
        out_c = torch.nn.functional.relu(self.fc1_c(out_c))
        out_c = self.fc2_c(out_c)
        out_c = out_c.view(-1,2)
        
        # Action
        bnac = self.BNet_action(action)
        bnac = bnac.view(bs,p,bnac.shape[-1])
        self.BiLstm_action.flatten_parameters()
        out_ac, (hac, cac) = self.BiLstm_action(bnac, None)
        out_ac = torch.nn.functional.relu(self.fc1_ac(out_ac))
        out_ac = self.fc2_ac(out_ac)
        out_ac = out_ac.view(-1,2)
        
        # Audio
        bnau = self.BNet_audio(audio)
        bnau = bnau.view(bs,p,bnau.shape[-1])
        self.BiLstm_audio.flatten_parameters()
        out_au, (hau, cau) = self.BiLstm_audio(bnau, None)
        out_au = torch.nn.functional.relu(self.fc1_au(out_au))
        out_au = self.fc2_au(out_au)
        out_au = out_au.view(-1,2)

        return out_p+out_c+out_ac+out_au
