from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

# Add UMI imports for coordinate conversion
from umi.real_world.real_inference_util import get_real_umi_action
from scipy.spatial.transform import Rotation as R
from . import eadp_utils

class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            action_pose_repr='relative',
            guided_steps: int = 15,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        # self.action_horizon = action_horizon # used for training
        self.action_horizon = 32 # TODO: get this action_horizon from MPC config
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Guidance parameters
        self.guidance = 0.0
        self.guided_steps = guided_steps
        self.env = None

        # Store for coordinate conversion and pregrad trajectory for visualization
        self.original_env_obs_stacked = None  # Store original env_obs_stacked before transformation
        self.action_pose_repr = action_pose_repr
        self.initial_rotation_matrix = None  # Store R_0 for gradient conversion
        self.initial_quaternion = None # Store initial quaternion for gradient conversion
        self.last_pregrad_trajectory = None  # Store pregrad trajectory in world coordinates

    # ========= MPC solver state management ============
    def _save_mpc_solver_state(self):
        # MPC solver
        WholeBodyMPC = self.env.unwrapped.controller
        self.mpc_planner = WholeBodyMPC.planners[0]
        solver = self.mpc_planner.mpc.ocp_solver
        """Save MPC solver state for warm-start restoration"""
        return {
            'x': [solver.get(k, "x").copy() for k in range(solver.N + 1)],
            'u': [solver.get(k, "u").copy() for k in range(solver.N)],
            'lam': [solver.get(k, "lam").copy() for k in range(solver.N + 1)],
            'pi': [solver.get(k, "pi").copy() for k in range(solver.N)]
        }

    def _restore_mpc_solver_state(self, state):
        """Restore MPC solver state from saved state"""
        if state is None:
            return
        # MPC solver
        WholeBodyMPC = self.env.unwrapped.controller
        self.mpc_planner = WholeBodyMPC.planners[0]
        solver = self.mpc_planner.mpc.ocp_solver
        for k in range(solver.N):
            solver.set(k, "x", state['x'][k])
            solver.set(k, "lam", state['lam'][k])
            solver.set(k, "pi", state['pi'][k])
        solver.set(solver.N, "x", state['x'][solver.N])
        solver.set(solver.N, "lam", state['lam'][solver.N])

    def _apply_mpc_gradient(self, trajectory, guidance_scale):
        """
        Apply MPC gradient to trajectory with scaling and normalization.
        
        Returns:
            tuple: (updated_trajectory, total_cost, tracking_cost)
        """
        # Convert to world coordinates
        traj_real = self.convert_trajectory_to_real_coordinates(trajectory)
        traj_np = traj_real[0].detach().cpu().numpy()  

        # Get MPC gradient
        grad_np = self.env.unwrapped.controller.tracking_gradient  # (H, 7)

        # Convert gradient back
        world_quats_tensor = traj_real[0, :, 3:7]
        grad = self.convert_gradient_to_umi_coordinates(grad_np, world_quats_tensor)
        
        # Apply gradient scaling with separate norms for pos and rot
        max_pos_norm = 0.05
        max_rot_norm = 0.005
        grad_pos = grad[:, :3]
        grad_rot = grad[:, 3:9]
        grad_gripper = grad[:, 9:]
        g_norm_pos = grad_pos.norm(dim=1, keepdim=True) + 1e-8
        g_norm_rot = grad_rot.norm(dim=1, keepdim=True) + 1e-8
        grad_pos = grad_pos * torch.clamp(max_pos_norm / g_norm_pos, max=1.0)
        grad_rot = grad_rot * torch.clamp(max_rot_norm / g_norm_rot, max=1.0)
        grad = torch.cat([grad_pos, grad_rot, grad_gripper], dim=1)

        return trajectory - guidance_scale * grad
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        # Initialize MPC solver state
        if self.guidance > 0.0:
            self._mpc_solver_initial_state = self._save_mpc_solver_state()

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]    

        # Store pregrad trajectory for visualization (before any refinement)
        # Convert to world coordinates using the same method as logging
        if self.guidance > 0.0:
            pregrad_world = self.convert_trajectory_to_real_coordinates(trajectory)
            pregrad_actions = pregrad_world[0].detach().cpu().numpy()  # (8, 8) full action
            self.last_pregrad_trajectory = pregrad_actions.copy()
        else:
            self.last_pregrad_trajectory = None

        if self.guidance > 0.0:
            # Get the smallest non-zero timestep for refinement
            smallest_nonzero_timestep = scheduler.timesteps[-2] if len(scheduler.timesteps) > 1 else scheduler.timesteps[0]
            
            for refinement_step in range(self.guided_steps):
                # Restore MPC solver state to avoid history effects
                # self._restore_mpc_solver_state(self._mpc_solver_initial_state)
                
                # Apply MPC gradient
                trajectory = self._apply_mpc_gradient(trajectory, self.guidance)

                # 6. Add noise back to smallest non-zero timestep (like training)
                noise = torch.randn(trajectory.shape, device=trajectory.device) / 4
                reduced_timestep = max(smallest_nonzero_timestep // 3, 0)
                noisy_trajectory = scheduler.add_noise(trajectory, noise, reduced_timestep)
                
                # 7. Apply conditioning to noisy trajectory
                noisy_trajectory[condition_mask] = condition_data[condition_mask]
                
                # 8. Single denoising step: smallest_timestep → 0
                # Set up scheduler for this specific denoising step
                scheduler.set_timesteps(2)  # Create sequence [smallest_nonzero_timestep, 0]
                # Manually override the timesteps to what we want
                scheduler.timesteps = torch.tensor([smallest_nonzero_timestep, 0], device=scheduler.timesteps.device)
                
                eps = model(noisy_trajectory, smallest_nonzero_timestep, 
                           local_cond=local_cond, global_cond=global_cond)
                out = scheduler.step(
                    eps, smallest_nonzero_timestep, noisy_trajectory, 
                    generator=generator,
                    **kwargs
                    )
                trajectory = out.prev_sample
                
                # 9. Apply conditioning after denoising
                trajectory[condition_mask] = condition_data[condition_mask]

        # Store minimal diffusion data
        traj_real_final = self.convert_trajectory_to_real_coordinates(trajectory)
        traj_np_final = traj_real_final[0].detach().cpu().numpy()  # (H, 8)
        
        # Store minimal data for tracking
        self.last_diffusion_data = {
            'final_trajectory': traj_np_final,  # (H, 8) - final output
        }

        return trajectory

    def plot_pre_vs_post_refinement(self, save_dir=None, tag="traj", show=False):
        """
        Visualize trajectory before refinement (pregrad) and after refinement (postgrad).
        Expects both as (H,8): [x,y,z,qw,qx,qy,qz,gripper]
        """
        # plot self.last_pregrad_trajectory 
        # plot self.last_diffusion_data['final_trajectory']
        if self.last_pregrad_trajectory is None or self.last_diffusion_data is None:
            return
        import matplotlib.pyplot as plt
        pregrad = self.last_pregrad_trajectory  # (H,8)
        postgrad = self.last_diffusion_data['final_trajectory']  # (H,8)
        H = pregrad.shape[0]
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        # set fixed axis limits 
        x_lims = [0, 1.8]
        y_lims = [-0.5, 0.5]
        z_lims = [0.9, 1.1]

        ax.set_zlim(z_lims)
        ax.plot(pregrad[:,0], pregrad[:,1], pregrad[:,2], 'r--', label='Pre-Refinement')
        ax.plot(postgrad[:,0], postgrad[:,1], postgrad[:,2], 'g-', label='Post-Refinement')
        ax.set_title('Trajectory Before and After MPC Refinement')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        if save_dir is not None:
            plt.savefig(f"{save_dir}/{tag}_pre_post_refinement.png")
        if show:
            plt.show()
        plt.close(fig)  

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        # self.plot_pre_vs_post_refinement(tag="mpc_refinement", show=True)
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= coordinate conversion ============
    def convert_trajectory_to_real_coordinates(self, trajectory):
        """
        Convert trajectory from normalized ee-frame to world coordinates
        Following the same coordinate-conversion pipeline used by the original ACT training stack.
        
        Args:
            trajectory: torch.Tensor of shape (batch_size, horizon, action_dim)
                       Currently in normalized coordinates
        
        Returns:
            trajectory: torch.Tensor in real world coordinates
        """            
        trajectory = self.normalizer['action'].unnormalize(trajectory)
        batch_size = trajectory.shape[0]
        converted_trajectories = []
        
        for b in range(batch_size):
            # Get trajectory for this batch item
            traj_b = trajectory[b]  # (action_horizon, action_dim)
            
            # Convert to numpy for processing
            raw_action = traj_b.detach().cpu().numpy()
            
            # Make positions relative to the first waypoint before the world-frame conversion.
            raw_action[...,:3] -= raw_action[0][:3]
            
            # Use the original env_obs_stacked that was stored during predict_action
            if self.original_env_obs_stacked is None:
                raise RuntimeError("original_env_obs_stacked is None. This should be set during predict_action.")
            
            # Store initial rotation matrix on first call for gradient conversion
            if self.initial_rotation_matrix is None:
                rot_repr = self.original_env_obs_stacked['robot0_eef_rot_axis_angle']
                if rot_repr.ndim > 1:
                    rot_repr = rot_repr[-1]  # Latest timestep
                # Handle 3D axis-angle rotation representation
                if rot_repr.shape[-1] == 3:
                    self.initial_rotation_matrix = R.from_rotvec(rot_repr).as_matrix()
                    # Also store initial quaternion (w, x, y, z) for gradient conversion
                    if self.initial_quaternion is None:
                        quat_xyzw_init = R.from_rotvec(rot_repr).as_quat()  # x,y,z,w
                        self.initial_quaternion = np.array([
                            quat_xyzw_init[3],  # w
                            quat_xyzw_init[0],  # x
                            quat_xyzw_init[1],  # y
                            quat_xyzw_init[2]   # z
                        ])
                else:
                    raise ValueError(f"Expected 3D axis-angle representation, got {rot_repr.shape[-1]}")
            
            # Convert to real coordinates using UMI's function
            action = get_real_umi_action(raw_action, self.original_env_obs_stacked, self.action_pose_repr)
            
            quat_xyzw = R.from_rotvec(action[...,3:6]).as_quat()
            quat_wxyz = np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:1], quat_xyzw[..., 1:2], quat_xyzw[..., 2:3]],axis=-1)  # (horizon, 4)

            # Assemble the final 8-D action
            action = np.concatenate(
                [action[..., 0:3],         # position (x, y, z)
                 quat_wxyz,               # quaternion (w, x, y, z)
                 action[..., 6:7]],       # gripper
                axis=-1
            )  # (horizon, 8)
            converted_trajectories.append(torch.from_numpy(action).to(trajectory.device))
        
        return torch.stack(converted_trajectories, dim=0)
    
    def convert_gradient_to_umi_coordinates(self, gradient, world_quats):
        """
        Convert gradients from world coordinates to UMI normalized coordinates
        
        Args:
            gradient: numpy.ndarray of shape (horizon, 7)
                     [pos_grad(3), quat_grad(4)] in world coordinates
        
        Returns:
            gradient: torch.Tensor of shape (horizon, 10) in normalized UMI coordinates
        """
        # 1. Convert numpy to torch and ensure correct device
        if isinstance(gradient, np.ndarray):
            gradient = torch.from_numpy(gradient).float()
        if isinstance(world_quats, np.ndarray):
            world_quats = torch.from_numpy(world_quats).float()

        gradient = gradient.to(self.device)
        # ensure dtype consistency (avoid Float/Double mismatch)
        world_quats = world_quats.to(self.device, dtype=gradient.dtype)

        H = gradient.shape[0]

        pos_grad_world  = gradient[:, :3]   # (H,3)
        quat_grad_world = gradient[:, 3:7]  # (H,4)

        if not hasattr(self, 'initial_quaternion') or self.initial_quaternion is None:
            raise RuntimeError("initial_quaternion not set – call convert_trajectory_to_real_coordinates first.")

        q0 = torch.from_numpy(self.initial_quaternion).float().to(self.device)  # (4,)
        w0, x0, y0, z0 = q0.unbind()

        L_q0 = torch.stack([
            torch.stack([ w0, -x0, -y0, -z0]),
            torch.stack([ x0,  w0, -z0,  y0]),
            torch.stack([ y0,  z0,  w0, -x0]),
            torch.stack([ z0, -y0,  x0,  w0])
        ], dim=0)  # (4,4)

        # batch multiply: g_q_rel = (L_q0^T) @ g_q_world  -> (H,4)
        g_q_rel = torch.einsum('ij,bj->bi', L_q0.T, quat_grad_world)

        # First compute relative quaternion q_rel = q0^{-1} ⊗ q_world
        # For unit quaternion, inverse is (w, -x, -y, -z)
        sign_vec = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q0.device)
        q0_conj = q0 * sign_vec  # (4,)

        # Quaternion product q_rel = q0_conj ⊗ q_world
        # Using Hamilton product in wxyz ordering
        w1, x1, y1, z1 = world_quats[:,0], world_quats[:,1], world_quats[:,2], world_quats[:,3]

        qr_w =  q0_conj[0]*w1 - q0_conj[1]*x1 - q0_conj[2]*y1 - q0_conj[3]*z1
        qr_x =  q0_conj[0]*x1 + q0_conj[1]*w1 + q0_conj[2]*z1 - q0_conj[3]*y1
        qr_y =  q0_conj[0]*y1 - q0_conj[1]*z1 + q0_conj[2]*w1 + q0_conj[3]*x1
        qr_z =  q0_conj[0]*z1 + q0_conj[1]*y1 - q0_conj[2]*x1 + q0_conj[3]*w1

        q_rel = torch.stack([qr_w, qr_x, qr_y, qr_z], dim=1)  # (H,4)

        w, x, y, z = q_rel[:,0], q_rel[:,1], q_rel[:,2], q_rel[:,3]

        # Build each row of J analytically (H,4)
        j0 = torch.stack([ torch.zeros_like(w), torch.zeros_like(w), -4*y, -4*z ], dim=1)
        j1 = torch.stack([  2*z,  2*y,  2*x,  2*w ], dim=1)
        j2 = torch.stack([ -2*y,  2*z, -2*w,  2*x ], dim=1)
        j3 = torch.stack([ -2*z,  2*y,  2*x, -2*w ], dim=1)
        j4 = torch.stack([ torch.zeros_like(w), -4*x, torch.zeros_like(w), -4*z ], dim=1)
        j5 = torch.stack([  2*x,  2*w,  2*z,  2*y ], dim=1)

        J = torch.stack([j0, j1, j2, j3, j4, j5], dim=1)   # (H,6,4)

        # g_rot6d = J @ g_q_rel
        g_rot6d = -torch.einsum('hij,hj->hi', J, g_q_rel)    # (H,6)

        # position gradient: rotate to robot frame
        R0 = torch.from_numpy(self.initial_rotation_matrix).float().to(self.device)  # (3,3)
        pos_grad_rel = pos_grad_world @ R0.T  # (H,3)

        # concat, normalise, return
        grad_10d = torch.cat([
            pos_grad_rel,
            g_rot6d,
            torch.zeros(H,1, device=self.device, dtype=gradient.dtype)
        ], dim=1)  # (H,10)

        grad_batch = grad_10d.unsqueeze(0)
        grad_norm = self.normalizer['action'].normalize(grad_batch).squeeze(0)

        # Zero out gripper dim 
        grad_norm[:,9:] = 0.0

        return grad_norm

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
