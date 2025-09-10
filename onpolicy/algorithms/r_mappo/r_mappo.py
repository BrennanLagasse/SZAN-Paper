from curses.panel import new_panel
import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm, IdentityTrans
from onpolicy.algorithms.utils.util import check
import math

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, policy, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.algo_name = args.algorithm_name

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.num_agents = args.num_agents
        self.max_z = args.max_z
        self.gamma = args.gamma
        self.div_thresh = args.div_thresh
        self.rex_thresh = args.rex_thresh
        self.alpha_rex_coeff = args.alpha_rex
        self.alpha_div_coeff = args.alpha_div
        self.sdpo_entropy_coeff = args.sdpo_entropy_coeff

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, \
            ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.ex_value_normalizer = self.policy.critic.v_out
            self.in_value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.ex_value_normalizer = ValueNorm(args, 1, device = self.device)
            self.in_value_normalizer = ValueNorm(args, 1, device = self.device)
        else:
            self.ex_value_normalizer = None
            self.in_value_normalizer = None
        
        self.env_num = 0


    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, value_norm, z_idxs):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        if self._use_popart or self._use_valuenorm:
            value_norm.update(return_batch, z_idxs)
            error_clipped = value_norm.normalize(return_batch) - value_pred_clipped
            error_original = value_norm.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, train_info, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        self.env_num += 1

        share_obs_batch, obs_batch, \
        rnn_states_batch, rnn_states_z_batch, loc_rnn_states_z_batch, \
        rnn_states_ex_critic_batch, rnn_states_in_critic_batch, \
        actions_batch, ex_value_preds_batch, in_value_preds_batch, \
        ex_return_batch, in_return_batch, masks_batch, active_masks_batch, \
        old_action_log_probs_batch, ex_adv_targ, in_adv_targ, available_actions_batch = sample

        ex_adv_targ = check(ex_adv_targ).to(**self.tpdv)
        in_adv_targ = check(in_adv_targ).to(**self.tpdv)
        ex_return_batch = check(ex_return_batch).to(**self.tpdv)
        in_return_batch = check(in_return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        ex_value_preds_batch = check(ex_value_preds_batch).to(**self.tpdv)
        in_value_preds_batch = check(in_value_preds_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)

        # z_idxs
        z_vec = obs_batch[:,:self.max_z].copy()
        z_idx = np.argmax(z_vec, -1)
        z_idxs = np.expand_dims(z_idx, -1)
        z_idxs_batch = check(z_idxs).to(**self.tpdv).long()

        # Reshape to do in a single forward pass for all steps
        ex_values, in_values, action_log_probs, dist_entropy = \
            self.policy.evaluate_actions(
                share_obs_batch,
                obs_batch, 
                rnn_states_batch, 
                rnn_states_ex_critic_batch, 
                rnn_states_in_critic_batch,
                actions_batch, 
                masks_batch, 
                available_actions_batch,
                active_masks_batch
            )
                                                                                    
        z_log_probs, _ = self.policy.evaluate_z(
            share_obs_batch, rnn_states_z_batch, masks_batch, active_masks=active_masks_batch
        )

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        ex_surr1 = imp_weights * ex_adv_targ 
        ex_surr2 = torch.clamp(imp_weights, 1.-self.clip_param, 1.+self.clip_param) * ex_adv_targ
        ex_L_clip = -torch.sum(torch.min(ex_surr1, ex_surr2), dim=-1, keepdim=True)

        in_surr1 = imp_weights * in_adv_targ 
        in_surr2 = torch.clamp(imp_weights, 1.-self.clip_param, 1.+self.clip_param) * in_adv_targ
        in_L_clip = -torch.sum(torch.min(in_surr1, in_surr2), dim=-1, keepdim=True)

        diver_mask = z_log_probs.detach() > math.log(self.div_thresh)

        cur_value = self.ex_value_normalizer.get_z_mean()
        Rex_mask = (cur_value > self.rex_thresh) # -50, -3
        Rex_mask = Rex_mask[z_idxs_batch]

        # print(f"MASK ANALYSIS DEBUG:")
        # print(f"  z_log_probs range: [{z_log_probs.min():.4f}, {z_log_probs.max():.4f}]")
        # print(f"  math.log(div_thresh): {math.log(self.div_thresh):.4f}")
        # print(f"  cur_value range: [{cur_value.min():.4f}, {cur_value.max():.4f}]")
        # print(f"  rex_thresh: {self.rex_thresh:.4f}")

        if self.algo_name == "MAPPO":
            target = ex_L_clip
        else:
            #i think this is the origonal, can confimr with repository if need to 
            # target = ex_L_clip * diver_mask 
            # target = target + in_L_clip * Rex_mask * (self.env_num>100) * self.alpha_rex_coeff
            # target = target + in_L_clip * ~diver_mask * self.alpha_div_coeff
            # # target = target - dist_entropy.unsqueeze(1) * ~diver_mask * self.sdpo_entropy_coeff 
            
            #this is the newer addition of the target calculation 
            # target = ex_L_clip + in_L_clip * self.alpha_div_coeff * 0.01  # Small diversity weight

            # Temporarily disable diversity entirely (this is newested addition, th eoriognal is above)
            if self.env_num < 500:  # First 500 updates = pure Pong learning
                target = ex_L_clip
            else:  # After 500 updates = gradual diversity introduction
                target = ex_L_clip + in_L_clip * self.alpha_div_coeff * 0.005




        policy_loss = target - dist_entropy.mean() * self.entropy_coef
        policy_loss = (policy_loss * active_masks_batch).sum() / active_masks_batch.sum()


        # Update your debug block to:
        # if self.env_num % 10 == 0:
        #     print(f"\n=== POLICY LOSS DEBUG (Update {self.env_num}) ===")
        #     print(f"COMPONENT MAGNITUDES:")
        #     print(f"  ex_L_clip mean: {ex_L_clip.mean():.6f}")
        #     print(f"  in_L_clip mean: {in_L_clip.mean():.6f}")
            
        #     # Fix the diversity contribution calculation
        #     if self.env_num < 500:
        #         diversity_contribution = 0.0  # No diversity for first 500 updates
        #     else:
        #         diversity_contribution = (in_L_clip * self.alpha_div_coeff * 0.005).mean()
            
        #     print(f"  diversity contribution: {diversity_contribution:.6f}")
        #     print(f"  entropy contribution: {(dist_entropy.mean() * self.entropy_coef):.6f}")
            
        #     print(f"FINAL VALUES:")
        #     print(f"  target (before entropy): {target.mean():.6f}")
        #     print(f"  policy_loss (final): {policy_loss:.6f}")
            
        #     print(f"RATIOS:")
        #     if abs(ex_L_clip.mean()) > 1e-8 and self.env_num >= 500:
        #         ratio = abs(diversity_contribution) / abs(ex_L_clip.mean())
        #         print(f"  diversity/extrinsic ratio: {ratio:.2f}")
        #     else:
        #         print(f"  diversity/extrinsic ratio: 0.00")
        #     print("=" * 50)





        self.policy.actor_optimizer.zero_grad()



        if update_actor:
            policy_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())


        if self.env_num % 50 == 0:  # Every 50 updates
            print(f"\n=== PPO LEARNING DIAGNOSTICS (Update {self.env_num}) ===")
            
            # Check if gradients are flowing
            total_norm = 0
            param_count = 0
            for p in self.policy.actor.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            total_norm = total_norm ** (1. / 2)
            
            print(f"GRADIENT HEALTH:")
            print(f"  Actor grad norm: {total_norm:.6f}")
            print(f"  Learning rate: {self.policy.actor_optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Policy loss magnitude: {abs(policy_loss.item()):.6f}")



        self.policy.actor_optimizer.step()

        # extrinsic critic update
        ex_value_loss = self.cal_value_loss(
            ex_values, ex_value_preds_batch, ex_return_batch, active_masks_batch, self.ex_value_normalizer, z_idxs
        )

        self.policy.ex_critic_optimizer.zero_grad()

        (ex_value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            ex_critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.ex_critic.parameters(), self.max_grad_norm)
        else:
            ex_critic_grad_norm = get_gard_norm(self.policy.ex_critic.parameters())

        self.policy.ex_critic_optimizer.step()

        # intrinsic critic update
        in_value_loss = self.cal_value_loss(
            in_values, in_value_preds_batch, in_return_batch, active_masks_batch, self.in_value_normalizer, z_idxs
        )

        self.policy.in_critic_optimizer.zero_grad()

        (in_value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            in_critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.in_critic.parameters(), self.max_grad_norm)
        else:
            in_critic_grad_norm = get_gard_norm(self.policy.in_critic.parameters())

        self.policy.in_critic_optimizer.step()


        if self.env_num % 50 == 0:
            print(f"VALUE FUNCTION DIAGNOSTICS:")
            print(f"  Ex value loss: {ex_value_loss.item():.6f}")
            print(f"  Ex values range: [{ex_values.min():.3f}, {ex_values.max():.3f}]")
            print(f"  Ex returns range: [{ex_return_batch.min():.3f}, {ex_return_batch.max():.3f}]")
            print(f"  Ex advantages range: [{ex_adv_targ.min():.3f}, {ex_adv_targ.max():.3f}]")
            print("=" * 50)


        # discriminator update
        z_log_probs, _ = self.policy.evaluate_z(
            share_obs_batch, rnn_states_z_batch, masks_batch, active_masks=active_masks_batch, isTrain=True
        )

        z_loss = -torch.mean(z_log_probs)
        self.policy.discri_optimizer.zero_grad()
        
        z_loss.backward()

        if self._use_max_grad_norm:
            z_grad_norm = nn.utils.clip_grad_norm_(self.policy.discriminator.parameters(), self.max_grad_norm)
        else:
            z_grad_norm = get_gard_norm(self.policy.discriminator.parameters())

        self.policy.discri_optimizer.step()

        train_info = dict()
        train_info['ex_value_loss'] = ex_value_loss
        train_info['in_value_loss'] = in_value_loss
        train_info['policy_loss'] = policy_loss
        train_info['dist_entropy'] = dist_entropy.mean()
        train_info['z_loss'] = z_loss
        train_info['imp_weight'] = imp_weights.mean()
        train_info['diver_mask'] = (diver_mask*1.).mean()
        train_info['Rex_mask'] = (Rex_mask*1.).mean()
        for z in range(self.max_z):
            train_info['cur_value_{}'.format(z)] = cur_value[z].mean()

        # print(f"TRAINING DEBUG:")
        # print(f"  diver_mask True%: {(diver_mask*1.).mean():.2f}")
        # print(f"  Rex_mask True%: {(Rex_mask*1.).mean():.2f}")
        # print(f"  alpha_rex_coeff: {self.alpha_rex_coeff}")
        # print(f"  alpha_div_coeff: {self.alpha_div_coeff}")
        # print(f"  env_num: {self.env_num}")

        return train_info

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        
        if self._use_popart or self._use_valuenorm:
            
            ex_total_return = buffer.ex_returns[:-1]
            ex_total_preds = self.ex_value_normalizer.denormalize(buffer.ex_value_preds[:-1]) 
            ex_advantages = ex_total_return - ex_total_preds

            in_total_return = buffer.in_returns[:-1]
            in_total_preds = self.in_value_normalizer.denormalize(buffer.in_value_preds[:-1]) 
            in_advantages = in_total_return - in_total_preds

        else:
            ex_total_return = buffer.ex_returns[:-1] 
            ex_total_preds = buffer.ex_value_preds[:-1] 
            ex_advantages = ex_total_return - ex_total_preds
            in_total_return = buffer.in_returns[:-1] 
            in_total_preds = buffer.in_value_preds[:-1] 
            in_advantages = in_total_return - in_total_preds

        ex_advantages_copy = ex_advantages.copy()
        ex_advantages_copy[buffer.active_masks[:-1]==0.0] = np.nan
        ex_mean_advantages = np.nanmean(ex_advantages_copy)
        ex_std_advantages = np.nanstd(ex_advantages_copy)
        ex_advantages = (ex_advantages - ex_mean_advantages) / (ex_std_advantages + 1e-5)

        in_advantages_copy = in_advantages.copy()
        in_advantages_copy[buffer.active_masks[:-1]==0.0] = np.nan
        in_mean_advantages = np.nanmean(in_advantages_copy)
        in_std_advantages = np.nanstd(in_advantages_copy)
        in_advantages = (in_advantages - in_mean_advantages) / (in_std_advantages + 1e-5)

        # print(f"TRAINER DEBUG - Advantage computation:")
        # print(f"  in_total_return range: [{in_total_return.min():.6f}, {in_total_return.max():.6f}]")
        # print(f"  in_total_preds range: [{in_total_preds.min():.6f}, {in_total_preds.max():.6f}]")
        # print(f"  in_advantages range: [{in_advantages.min():.6f}, {in_advantages.max():.6f}]")
        # print(f"  in_mean_advantages: {in_mean_advantages:.6f}")
        # print(f"  in_std_advantages: {in_std_advantages:.6f}")
        

        train_info = {}

        for epoch in range(self.ppo_epoch):

            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    ex_advantages, 
                    in_advantages, 
                    self.num_mini_batch, 
                    self.data_chunk_length
                )
            elif self._use_naive_recurrent:
                raise NotImplementedError
            else:
                raise NotImplementedError

            for sample in data_generator:

                info = self.ppo_update(sample, update_actor)

                for key in info:
                    if key in train_info:
                        train_info[key] += info[key]
                    else:
                        train_info[key] = info[key]

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.ex_critic.train()
        self.policy.in_critic.train()
        self.policy.discriminator.train()
        self.policy.local_discri.train()
        self.policy.alpha_model.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.ex_critic.eval()
        self.policy.in_critic.train()
        self.policy.discriminator.eval()
        self.policy.local_discri.eval()
        self.policy.alpha_model.eval()
