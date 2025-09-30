from .base import BaseSampler


class DDPMSampler(BaseSampler):
    def __init__(self, num_steps=50, beta_start=0.0001, beta_end=0.5, schedule="quad", device="cuda:0"):
        super().__init__(num_steps, beta_start, beta_end, schedule, device)

        self.alpha_bar_sqrt = self.alpha_bar ** 0.5
        self.one_minus_alpha_bar_sqrt = (1 - self.alpha_bar)**0.5

        self.reverse_coef1 = 1/self.alpha**0.5
        self.reverse_coef2 = (1 - self.alpha)/self.one_minus_alpha_bar_sqrt
        
        numerator = 1 - self.alpha_bar[:-1]
        denomitor = 1 - self.alpha_bar[1:]
        sigma_sq = numerator/denomitor * self.beta[1:]
        self.sigma = sigma_sq ** 0.5

    def forward(self, x0, t, noise):
        """
        Map the real time series x into noise.
        Args:
            x0: (B,V,L) time series in the real space.
            t: (B) diffusion steps.
            noise: (B,V,L) input noise
        Return:
            noisy_x: (B,V,L)
        """
        noisy_x = self.alpha_bar_sqrt[t] * x0 + self.one_minus_alpha_bar_sqrt[t] * noise
        return noisy_x
    
    def reverse(self, xt, pred_noise, t, noise):
        """
        Decode the noisy xt into x_prev based on side information and attributes.
        Args:
            xt: (B,V,L) noisy time series at the diffusion step t.
            pred_noise: (B,V,L)
            t: the diffusion step. A number within [self.num_steps-1, 0].
            noise: (B,V,L)
        """
        coef1 = self.reverse_coef1[t]
        coef2 = self.reverse_coef2[t]
        x_prev = coef1 * (xt - coef2*pred_noise)

        mask = (t > 0).unsqueeze(-1).unsqueeze(-1).float()
        x_prev += mask*(self.sigma[t-1] * noise)
        return x_prev
