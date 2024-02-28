import torch
import numpy as np
import copy

class wireless_channel():

    def __init__(self,args) -> None:
        self.num_tx = args.num_tx
        self.num_rx = args.num_rx
        self.args = args
        self.device = 'cuda' if args.gpu else 'cpu'
    
    def generate_channel_gain(self,num_tx,num_rx,device):
        '''
        Return Rayleight distributed channel gain coefficient, dimension = num_tx x num_rx
        '''
        # Rayleigh parameters
        average_power_loss = 1e-3  # Average power loss (10^(-3))

        # Calculate the Rayleigh scale parameter (sigma)
        # The scale parameter is related to the average power loss as follows:
        # average_power_loss = 2 * sigma^2
        sigma = np.sqrt(average_power_loss / 2)

        # Generate independent Rayleigh fading samples
        rayleigh_samples = sigma * np.random.randn(num_tx,num_rx) + 1j * sigma * np.random.randn(num_tx,num_rx)
        rayleigh_samples = torch.from_numpy(rayleigh_samples).to(device)
        return rayleigh_samples

    def generate_channel_noise(self,signal_tensor:torch.Tensor,device,snr_db=-30):
        '''
        Return Noise tensor with same shape and device like signal tensor
        '''
        num_samples = signal_tensor.numel()
        sigma = np.power(10,snr_db/10.0)
        noise = sigma * np.random.randn(num_samples)
        noise = torch.from_numpy(noise).to(device)
        noise = noise.reshape_as(signal_tensor)
        return noise

    def channel_process(self,w):
        '''
        Apply MIMO channel porcess (precode, over the air, postcode), return added weight

        w: list of all clients' weights
        '''
        device = 'cuda' if self.args.gpu else 'cpu'
        if self.num_tx == 1: 
            rayleigh_coefficient = self.generate_channel_gain(len(w),self.num_rx,device)       
            w_avg = copy.deepcopy(w[0])
            for key in w_avg.keys():
                for i in range(len(w)):  # Each cients' SIMO channel
                    # Put your precoding here:
                    precode = 1/rayleigh_coefficient    # Only can Use precode[i,0] since each client have 1 attenna
                    # Put your postcoding here:
                    postcode = torch.ones(self.num_rx).to(device)                
                    # yij = hij*gij*x + ni, yi = real(Σ pj*yij)

                    # attenna 0
                    y_complex = rayleigh_coefficient[i,0] * precode[i,0] * w[i][key] + self.generate_channel_noise(w[i][key],device,self.args.snr_db)
                    y_post_complex = postcode[0] * y_complex
                    # attenna 1-X
                    for rx in range(1,self.num_rx):
                        y_complex = rayleigh_coefficient[i,rx] * precode[i,0] * w[i][key] + self.generate_channel_noise(w[i][key],device,self.args.snr_db)
                        # 全部用precode[i,0]需要改进
                        y_post_complex += postcode[rx] * y_complex
                    if i==0:
                        w_avg[key] = torch.real(y_post_complex)
                    else:
                        w_avg[key] += torch.real(y_post_complex)
                w_avg[key] = torch.div(w_avg[key], len(w)*self.num_rx)

        return w_avg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--num_tx', type=int, default=1)
    parser.add_argument('--num_rx', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--snr_db', type=int, default=-30)
    args = parser.parse_args()

    channel = wireless_channel(args)
    model = {'layer1':torch.Tensor([1,2,-1]).to('cuda')}
    model_list = [model,model,model]
    print(channel.channel_process(model_list))