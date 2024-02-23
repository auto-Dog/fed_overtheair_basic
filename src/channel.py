import torch
import numpy as np
import copy

def generate_channel_gain(num_users,device):
    # Rayleigh parameters
    average_power_loss = 1e-3  # Average power loss (10^(-3))
    num_samples = num_users  # Number of fading samples to generate

    # Calculate the Rayleigh scale parameter (sigma)
    # The scale parameter is related to the average power loss as follows:
    # average_power_loss = 2 * sigma^2
    sigma = np.sqrt(average_power_loss / 2)

    # Generate independent Rayleigh fading samples
    rayleigh_samples = sigma * np.random.randn(num_samples) + 1j * sigma * np.random.randn(num_samples)
    rayleigh_samples = torch.from_numpy(rayleigh_samples).to(device)
    return rayleigh_samples

def generate_channel_noise(signal_tensor:torch.Tensor,device,snr_db=-30):
    num_samples = signal_tensor.numel()
    sigma = np.power(10,snr_db/10.0)
    noise = sigma * np.random.randn(num_samples)
    noise = torch.from_numpy(noise).to(device)
    noise = noise.reshape_as(signal_tensor)
    return noise

def channel_process(w,args):
    device = 'cuda' if args.gpu else 'cpu'
    rayleigh_coefficient = generate_channel_gain(len(w),device)
    # Put your precoding here:
    precode = 1/rayleigh_coefficient
    # Put your postcoding here:
    postcode = {}
    for key in w[0]:
        postcode[key] = torch.ones_like(w[0][key]).to(device)
    
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            # yi = hi*gi*x + n
            y_complex = rayleigh_coefficient[i] * precode[i] * w[i][key] + generate_channel_noise(w[i][key],device,args.snr_db)
            w_avg[key] += torch.real(y_complex)
        # y = p*sum(yi)
        w_avg[key] = torch.div(w_avg[key], len(w))
        w_avg[key] = w_avg[key] * postcode[key]

    return w_avg
