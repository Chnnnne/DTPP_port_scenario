import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)


class AgentEncoder(nn.Module):
    def __init__(self, agent_dim):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output
    

class VectorMapEncoder(nn.Module):
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.position_encode = PositionalEncoding(max_len=map_len)

    def segment_map(self, map, map_encoding):
        B, N_e, N_p, D = map_encoding.shape 
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        output = self.position_encode(self.point_net(input))
        encoding, mask = self.segment_map(input, output)

        return encoding, mask


class CrossAttention(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, attn_mask=mask)
        attention_output = self.norm_1(attention_output)
        linear_output = self.ffn(attention_output)
        output = attention_output + self.dropout(linear_output)
        output = self.norm_2(output)

        return output


class AgentDecoder(nn.Module):
    def __init__(self, max_time, max_branch, dim):
        super(AgentDecoder, self).__init__()
        self._max_time = max_time
        self._max_branch = max_branch
        self.traj_decoder = nn.Sequential(nn.Linear(dim, 128), nn.ELU(), nn.Linear(128, 3*10))

    def forward(self, encoding, current_state):
        encoding = torch.reshape(encoding, (encoding.shape[0], self._max_branch, self._max_time, 512))
        agent_traj = self.traj_decoder(encoding).reshape(encoding.shape[0], self._max_branch, self._max_time*10, 3)
        agent_traj += current_state[:, None, None, :3]

        return agent_traj
    

class ScoreDecoder(nn.Module):
    def __init__(self, variable_cost=False):
        super(ScoreDecoder, self).__init__()
        self._n_latent_features = 4
        self._variable_cost = variable_cost

        self.interaction_feature_encoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 256))
        self.interaction_feature_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self._n_latent_features), nn.Sigmoid())
        self.weights_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self._n_latent_features+4), nn.Softplus())

    def get_hardcoded_features(self, ego_traj, max_time):
        '''将ego的M条规划轨迹（时间步为T）提取特征到4维， output:B,M,4 '''
        # ego_traj: B, M, T, 6
        # x, y, yaw, v, a, r曲率

        speed = ego_traj[:, :, :max_time, 3] # B,M,T
        acceleration = ego_traj[:, :, :max_time, 4] # B,M,T
        jerk = torch.diff(acceleration, dim=-1) / 0.1 # B,M,T-1
        jerk = torch.cat((jerk[:, :, :1], jerk), dim=-1)# B,M,T
        curvature = ego_traj[:, :, :max_time, 5] # B,M,T
        lateral_acceleration = speed ** 2 * curvature # B,M,T

        speed = -speed.mean(-1).clip(0, 15) / 15 # B,M, 平均速度取反
        acceleration = acceleration.abs().mean(-1).clip(0, 4) / 4 # B, M 平均加速度
        jerk = jerk.abs().mean(-1).clip(0, 6) / 6 # B,M 平均加加速度
        lateral_acceleration = lateral_acceleration.abs().mean(-1).clip(0, 5) / 5 # B,M 平均横向加速度

        features = torch.stack((speed, acceleration, jerk, lateral_acceleration), dim=-1) # B,M,4

        return features
    
    def calculate_collision(self, ego_traj, agent_traj, agents_states, max_time):
        # ego_traj: B, T, 6
        # agent_traj: B, N, T, 3
        # agents_states: B, N, 11

        agent_mask = torch.ne(agents_states.sum(-1), 0) # B, N

        # Compute the distance between the two agents
        dist = torch.norm(ego_traj[:, None, :max_time, :2] - agent_traj[:, :, :max_time, :2], dim=-1) # B, 1, 50, 2 - B,N,50,2 -> B,N,50
    
        # Compute the collision cost
        cost = torch.exp(-0.2 * dist ** 2) * agent_mask[:, :, None] # B,N,50
        cost = cost.sum(-1).sum(-1)# B

        return cost
    
    def get_latent_interaction_features(self, ego_traj, agent_traj, agents_states, max_time):
        # ego_traj: B, T, 6
        # agent_traj: B, N, T, 3            N=_neighbors
        # agents_states: B, N, 11

        # Get agent mask
        agent_mask = torch.ne(agents_states.sum(-1), 0) # B, N

        # Get relative attributes of agents
        relative_yaw = agent_traj[:, :, :max_time, 2] - ego_traj[:, None, :max_time, 2]#B,N,T, - B,1,T,  计算一个ego的一个轨迹和N个agent的1个轨迹在T时间区间内的相对角度
        relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw))
        relative_pos = agent_traj[:, :, :max_time, :2] - ego_traj[:, None, :max_time, :2] # B,N,T,2 - B,1,T,2 计算一个ego的一个轨迹和N个agent的1个轨迹在T时间区间内的相对位置
        relative_pos = torch.stack([relative_pos[..., 0] * torch.cos(relative_yaw), 
                                    relative_pos[..., 1] * torch.sin(relative_yaw)], dim=-1)
        agent_velocity = torch.diff(agent_traj[:, :, :max_time, :2], dim=-2) / 0.1 # B,N,T-1, 2
        agent_velocity = torch.cat((agent_velocity[:, :, :1, :], agent_velocity), dim=-2) # B,N,T,2
        ego_velocity_x = ego_traj[:, :max_time, 3] * torch.cos(ego_traj[:, :max_time, 2])#BT*BT
        ego_velocity_y = ego_traj[:, :max_time, 3] * torch.sin(ego_traj[:, :max_time, 2])
        relative_velocity = torch.stack([(agent_velocity[..., 0] - ego_velocity_x[:, None]) * torch.cos(relative_yaw), # stack(BNT-B1T)
                                         (agent_velocity[..., 1] - ego_velocity_y[:, None]) * torch.sin(relative_yaw)], dim=-1)  # BNT2 得到一个ego和N个agent的1个轨迹在T时间区间内的相对速度
        relative_attributes = torch.cat((relative_pos, relative_yaw.unsqueeze(-1), relative_velocity), dim=-1) # B, N, T, 5得到1个ego的一个轨迹和N个agent的1个轨迹在T区间的相对特征

        # Get agent attributes
        agent_attributes = agents_states[:, :, None, 6:].expand(-1, -1, relative_attributes.shape[2], -1) # B, N, 11-> B,N,1,5->B,N,T,5
        attributes = torch.cat((relative_attributes, agent_attributes), dim=-1) # B,N,T,5 + 5 将1个ego的一个轨迹和N个agent的1个轨迹在T区间的相对特征
        attributes = attributes * agent_mask[:, :, None, None]# B,N,T,10* B,N,1,1

        # Encode relative attributes and decode to latent interaction features
        features = self.interaction_feature_encoder(attributes) # B,N,T,256
        features = features.max(1).values.mean(1)# B,N,T,256 -> B,T,256-> B,256 求所有agent的最大的，时间步平均的feature
        features = self.interaction_feature_decoder(features) # B,4
  
        return features

    def forward(self, ego_traj, ego_encoding, agents_traj, agents_states, timesteps):
        '''
        M条规划轨迹
        - ego_traj: B,M,T,6
        - ego_encoding: B, 256 
        - agents_traj: B, M, N, T, 3 缺点没有考虑多模态性
        - agents_states: B, N, 11
        计算

        weights:  B,8 用来权衡 hardcode和interaction的feature权重得到最终的scores
        一个交互场景对应一个score和collision_feature
        '''
        ego_traj_features = self.get_hardcoded_features(ego_traj, timesteps) # B,M,4   获取自车规划轨迹的一些编码特征
        if not self._variable_cost:
            ego_encoding = torch.ones_like(ego_encoding)
        weights = self.weights_decoder(ego_encoding) # B,8 用来权衡 hardcode和interaction的score权重
        ego_mask = torch.ne(ego_traj.sum(-1).sum(-1), 0) # B,M

        scores = []
        for i in range(agents_traj.shape[1]): # M，每一条规划轨迹 BT6  BNT3
            hardcoded_features = ego_traj_features[:, i] # B,4
            interaction_features = self.get_latent_interaction_features(ego_traj[:, i], agents_traj[:, i], agents_states, timesteps)#B,3m,4
            features = torch.cat((hardcoded_features, interaction_features), dim=-1)#B,4 + B,4 = B,8
            score = -torch.sum(features * weights, dim=-1)# B
            collision_feature = self.calculate_collision(ego_traj[:, i], agents_traj[:, i], agents_states, timesteps)# B
            score += -10 * collision_feature 
            scores.append(score) # M [B,]

        scores = torch.stack(scores, dim=1) # B, M
        scores = torch.where(ego_mask, scores, float('-inf'))

        return scores, weights