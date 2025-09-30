from .base import BaseSampler


class DDIMSampler(BaseSampler):
    def __init__(self, num_steps=50, beta_start=0.0001, beta_end=0.5, schedule="quad", device="cuda:0"):
        super().__init__(num_steps, beta_start, beta_end, schedule, device)

        self.alpha_bar_sqrt = self.alpha_bar**0.5
        self.alpha_bar_sqrt_inverse = 1/self.alpha_bar_sqrt
        self.one_minus_alpha_bar_sqrt = (1-self.alpha_bar)**0.5

        numerator = 1 - self.alpha_bar[:-1]
        denominator = 1 - self.alpha_bar[1:]
        sigma_sq = numerator/denominator * self.beta[1:]
        self.sigma = sigma_sq**0.5
        self.reverse_coef2 = (1-self.alpha_bar[:-1]-sigma_sq)**0.5
        self.reverse_coef2_determin = (1-self.alpha_bar[:-1])**0.5

    def forward(self, xt, pred_noise, t):
        """
        Eq.12 in DiffusionCLIP.
        """
        pred_x0 = self.predict_x0(xt, pred_noise, t)

        coef1 = self.alpha_bar_sqrt[t+1]
        coef2 = self.one_minus_alpha_bar_sqrt[t+1]
        x_next = coef1*pred_x0 + coef2*pred_noise
        return x_next
    
    def reverse(self, xt, pred_noise, t, noise, is_determin=False):
        """
        Eq.12 in the DDIM paper.
        """
        pred_x0 = self.predict_x0(xt, pred_noise, t)
        
        mask = (t == 0).unsqueeze(-1).unsqueeze(-1)

        coef1 = self.alpha_bar_sqrt[t-1]
        
        if is_determin:
            coef2 = self.reverse_coef2_determin[t-1]
            coef3 = 0
        else:
            coef2 = self.reverse_coef2[t-1]
            coef3 = self.sigma[t-1]
        x_prev = coef1*pred_x0 + coef2*pred_noise + coef3*noise
        x_prev = mask*pred_x0 + (~mask)*x_prev
        return x_prev

    def predict_x0(self, xt, pred_noise, t):
        """
        Eq.9 in the DDIM paper.
        Args:
            xt: (B,V,L) noisy time sieres.
            pred_noise: (B,V,L) predicted noise.
            t: (B) the diffusion step. A number within [self.num_steps-1, 0].
        Return:
            pred_x0: (B,V,L) predicted x0
        """
        mask = (t == -1).unsqueeze(-1).unsqueeze(-1).float()
        coef1 = self.one_minus_alpha_bar_sqrt[t]
        coef2 = self.alpha_bar_sqrt_inverse[t]
        pred_x0 = (xt - coef1*pred_noise) * coef2
        pred_x0 = mask*xt + (1-mask)*pred_x0
        return pred_x0
