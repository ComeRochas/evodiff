import torch
from evodiff.utils import Tokenizer
import pathlib
from tqdm import tqdm

from evodiff.pretrained import CARP_38M, CARP_640M, D3PM_BLOSUM_38M, D3PM_BLOSUM_640M, D3PM_UNIFORM_38M, D3PM_UNIFORM_640M,\
                           OA_DM_640M, OA_DM_38M, LR_AR_38M, LR_AR_640M, ESM1b_650M


home = str(pathlib.Path.home())

def prefix_reward_mstq(seq_str: str, target: str = 'MSTQ') -> float:
    """
    Reward on protein sequences defined as 4 - d(x[:4], 'MSTQ'),
    where d is the Hamming distance between equal-length prefixes.

    If the sequence is shorter than 4, compare only on its length.
    """
    if seq_str is None:
        return 0.0
    L = min(4, len(seq_str))
    if L == 0:
        return 0.0
    d = 0
    for i in range(L):
        c = seq_str[i]
        if c not in "ACDEFGHIKLMNPQRSTVWY":  # treat specials as mismatch
            d += 1
        elif c != target[i]:
            d += 1
    return float(4 - d)

def batch_prefix_rewards(token_batch: torch.Tensor, tokenizer: Tokenizer) -> torch.Tensor:
    """
    Compute rewards for a batch of tokenized sequences using prefix_reward_mstq.
    token_batch: [B, L] integer tokens
    returns: [B] rewards as float tensor
    """
    rewards = []
    with torch.no_grad():
        for s in token_batch:
            seq = tokenizer.untokenize(s)
            rewards.append(prefix_reward_mstq(seq))
    return torch.tensor(rewards, dtype=torch.float32, device=token_batch.device)



def generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, batch_size=3, device='cuda'):
    """
    Generate a random start string from uniform dist and convert to predictions
    """
    #model.eval()
    #device = model.device()

    sample = torch.randint(0, tokenizer.K, (batch_size, seq_len))
    sample = sample.to(torch.long)
    sample = sample.to(device)
    Q = Q.to(device)
    Q_bar = Q_bar.to(device)

    timesteps = torch.linspace(timesteps-1,1,int((timesteps-1)/1), dtype=int) # iterate over reverse timesteps
    timesteps = timesteps.to(device)
    with torch.no_grad():
        for t in tqdm(timesteps):
            timesteps = torch.tensor([t] * batch_size)
            timesteps = timesteps.to(device)
            prediction = model(sample, timesteps)
            p = prediction[:, :, :tokenizer.K]  # p_theta_tilde (x_0_tilde | x_t) # Don't predict non-standard AAs
            p = torch.nn.functional.softmax(p, dim=-1)  # softmax over categorical probs
            p = p.to(torch.float64)
            x_tminus1 = sample.clone()
            for i, s in enumerate(sample):
                x_t_b = tokenizer.one_hot(s)
                A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)  # [ P x K x K]
                B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred)  # [ P x K x K ]
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1,2),  p[i].unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
                # On final timestep pick next best from standard AA
                if t == 1:
                     x_tminus1[i] = torch.multinomial(p_theta_marg[:, :tokenizer.K-6], num_samples=1).squeeze()
                # diff = torch.ne(s, x_tminus1[i])
                # if t % 100 == 0:
                #     print("time", t, diff.sum().item(), "mutations", tokenizer.untokenize(x_tminus1[i]), "sample", tokenizer.untokenize(s))
            sample = x_tminus1

    untokenized = [tokenizer.untokenize(s) for s in sample]
    print("final seq", untokenized)
    return sample, untokenized

def generate_d3pm_smc(
        model,
        tokenizer,
        Q,
        Q_bar,
        timesteps,
        seq_len,
        batch_size: int = 30,
        device: str = 'cuda',
        reward_scale: float = 1.0,
        smc_every: int = 1,
        resample_strategy: str = 'multinomial',
):
        """
        D3PM generation with Sequential Monte Carlo (SMC) guidance.

        - Maintains N=batch_size particles.
        - At each reverse timestep, samples x_{t-1} ~ p_θ(x_{t-1}|x_t).
        - Applies a simple prefix reward at frequency `smc_every` steps and resamples particles
            with weights w_i ∝ exp(reward_scale * reward_i).

        Args:
            model, tokenizer, Q, Q_bar, timesteps: from pretrained D3PM_* (return_all=True)
            seq_len: target sequence length
            batch_size: number of particles
            device: 'cuda' or 'cpu'
            reward_scale: scale factor (lambda) for exp weighting; 0 disables guidance
            smc_every: apply SMC every k steps; 1 = every step
            resample_strategy: currently 'multinomial' only

        Returns:
            sample: [B, L] token tensor of final sequences
            untokenized: list[str] sequences
            rewards: [B] final rewards
        """
        sample = torch.randint(0, tokenizer.K, (batch_size, seq_len))
        sample = sample.to(torch.long).to(device)
        Q = Q.to(device)
        Q_bar = Q_bar.to(device)

        # Reverse diffusion timesteps (t from T-1 down to 1)
        ts = torch.linspace(timesteps - 1, 1, int((timesteps - 1) / 1), dtype=torch.int64, device=device)
        with torch.no_grad():
                for step_idx, t in enumerate(tqdm(ts)):
                        timesteps_tensor = torch.full((batch_size,), int(t.item()), dtype=torch.long, device=device)
                        prediction = model(sample, timesteps_tensor)
                        p = prediction[:, :, :tokenizer.K]
                        p = torch.nn.functional.softmax(p, dim=-1).to(torch.float64)

                        # Standard D3PM ancestral step
                        x_tminus1 = sample.clone()
                        for i, s in enumerate(sample):
                                x_t_b = tokenizer.one_hot(s)
                                A = torch.mm(x_t_b, torch.t(Q[t]))  # [L x K]
                                Q_expand = Q_bar[t - 1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)
                                B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                                q_t = torch.mul(A.unsqueeze(1), B_pred)  # [L x K x K]
                                p_theta_marg = torch.bmm(torch.transpose(q_t, 1, 2), p[i].unsqueeze(2)).squeeze()
                                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                                x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
                                if t == 1:
                                        x_tminus1[i] = torch.multinomial(p_theta_marg[:, :tokenizer.K - 6], num_samples=1).squeeze()
                        sample = x_tminus1

                        # SMC resampling step
                        # Apply at frequency smc_every, and only if reward_scale > 0
                        if reward_scale > 0 and smc_every > 0 and ((step_idx + 1) % smc_every == 0):
                                # Compute reward per particle based on current decoded sequences
                                rewards = batch_prefix_rewards(sample, tokenizer)  # [B]
                                # Importance weights proportional to exp(lambda * reward)
                                weights = torch.softmax(reward_scale * rewards, dim=0)
                                # Resample indices with replacement
                                if resample_strategy == 'multinomial':
                                        idx = torch.multinomial(weights, num_samples=batch_size, replacement=True)
                                else: #not implemented here
                                        idx = torch.multinomial(weights, num_samples=batch_size, replacement=True)
                                sample = sample[idx]

        untokenized = [tokenizer.untokenize(s) for s in sample]
        final_rewards = batch_prefix_rewards(sample, tokenizer)
        print("final seq (SMC)", untokenized)
        return sample, untokenized, final_rewards
