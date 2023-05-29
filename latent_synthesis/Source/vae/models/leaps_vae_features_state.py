from typing import Union
import numpy as np
import torch
import torch.nn as nn

from dsl import DSL
from karel.world import World
from karel.world_batch import WorldBatch

from ..utils import init_gru
from .leaps_vae import LeapsVAE
from .base_vae import BaseVAE, ModelOutput


class LeapsVAEFeaturesState(LeapsVAE):
    """Program VAE class used in LEAPS paper.
    """    
    def __init__(self, dsl: DSL, device: torch.device, hidden_size: Union[int, None] = None):
        super().__init__(dsl, device, hidden_size)
        
        self.num_state_features = len(dsl.get_features())
        
        # Input: enc(rho_i) (T), z (Z). Output: prog_out (Z).
        self.policy_gru = nn.GRU(self.hidden_size + self.num_state_features + self.num_agent_actions, self.hidden_size)
        init_gru(self.policy_gru)
        
        self.to(self.device)
    
    def get_states_features(self, states: torch.Tensor, teacher_enforcing = True):
        if not teacher_enforcing:
            features = self._world.get_all_features()
        else:
            states_np = states.detach().cpu().numpy().astype(np.bool_)
            # C x H x W to H x W x C
            states_np = np.moveaxis(states_np,[-1,-2,-3], [-2,-3,-1])
            world_batch = WorldBatch(states_np)
            features = world_batch.get_all_features()
        return torch.tensor(features, dtype=torch.bool, device=self.device)
    
    def policy_executor(self, z: torch.Tensor, s_h: torch.Tensor, a_h: torch.Tensor,
                        a_h_mask: torch.Tensor, a_h_teacher_enforcing = True):
        batch_size, demos_per_program, _, c, h, w = s_h.shape
        
        # Taking only first state and squeezing over first 2 dimensions
        current_state = s_h[:, :, 0, :, :, :].view(batch_size*demos_per_program, c, h, w)
        
        ones = torch.ones((batch_size*demos_per_program, 1), dtype=torch.long, device=self.device)
        current_action = (self.num_agent_actions - 1) * ones
        
        z_repeated = z.unsqueeze(1).repeat(1, demos_per_program, 1)
        z_repeated = z_repeated.view(batch_size*demos_per_program, self.hidden_size)
        
        gru_hidden = z_repeated.unsqueeze(0)
        
        pred_a_h = []
        pred_a_h_logits = []
        
        if not a_h_teacher_enforcing:
            self.env_init(current_state)
        
        current_features = self.get_states_features(current_state, a_h_teacher_enforcing)
        
        terminated_policy = torch.zeros_like(current_action, dtype=torch.bool, device=self.device)
        
        mask_valid_actions = torch.tensor((self.num_agent_actions - 1) * [-torch.finfo(torch.float32).max]
                                          + [0.], device=self.device)
        
        for i in range(1, self.max_demo_length):
            enc_action = self.action_encoder(current_action.squeeze(-1))
            
            gru_inputs = torch.cat((z_repeated, current_features, enc_action), dim=-1)
            gru_inputs = gru_inputs.unsqueeze(0)
            
            gru_out, gru_hidden = self.policy_gru(gru_inputs, gru_hidden)
            gru_out = gru_out.squeeze(0)
            
            pred_action_logits = self.policy_mlp(gru_out)
            
            masked_action_logits = pred_action_logits + terminated_policy * mask_valid_actions
            
            current_action = self.softmax(masked_action_logits).argmax(dim=-1).view(-1, 1)
            
            pred_a_h.append(current_action)
            pred_a_h_logits.append(pred_action_logits)
            
            # Apply teacher enforcing while training
            if a_h_teacher_enforcing:
                current_state = s_h[:, :, i, :, :, :].view(batch_size*demos_per_program, c, h, w)
                current_action = a_h[:, :, i].view(batch_size*demos_per_program, 1)
            # Otherwise, step in actual environment to get next state
            else:
                current_state = self.env_step(current_state, current_action)
                
            current_features = self.get_states_features(current_state, a_h_teacher_enforcing)
                
            terminated_policy = torch.logical_or(current_action == self.num_agent_actions - 1,
                                                 terminated_policy)
    
        pred_a_h = torch.stack(pred_a_h, dim=1).squeeze(-1)
        pred_a_h_logits = torch.stack(pred_a_h_logits, dim=1)
        pred_a_h_masks = (pred_a_h != self.num_agent_actions - 1)
        
        return pred_a_h, pred_a_h_logits, pred_a_h_masks
    
    def forward(self, s_h: torch.Tensor, a_h: torch.Tensor, a_h_mask: torch.Tensor, 
                prog: torch.Tensor, prog_mask: torch.Tensor, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelOutput:
        z = self.encode(prog, prog_mask)
        
        decoder_result = self.decode(z, prog, prog_mask, prog_teacher_enforcing)
        pred_progs, pred_progs_logits, pred_progs_masks = decoder_result
        
        policy_result = self.policy_executor(z, s_h, a_h, a_h_mask, a_h_teacher_enforcing)
        pred_a_h, pred_a_h_logits, pred_a_h_masks = policy_result
        
        return ModelOutput(pred_progs, pred_progs_logits, pred_progs_masks,
                           pred_a_h, pred_a_h_logits, pred_a_h_masks)
        
    def encode_program(self, prog: torch.Tensor):
        if prog.dim() == 1:
            prog = prog.unsqueeze(0)
        
        prog_mask = (prog != self.num_program_tokens - 1)
        
        z = self.encode(prog, prog_mask)
        
        return z
    
    def decode_vector(self, z: torch.Tensor):
        pred_progs, _, pred_progs_masks = self.decode(z, None, None, False)
        
        pred_progs_tokens = []
        for prog, prog_mask in zip(pred_progs, pred_progs_masks):
            pred_progs_tokens.append([0] + prog[prog_mask].cpu().numpy().tolist())
        
        return pred_progs_tokens