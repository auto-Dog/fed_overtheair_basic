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
        average_power_loss = 1e-2  # Average power loss (10^(-3))

        # Calculate the Rayleigh scale parameter (sigma)
        # The scale parameter is related to the average power loss as follows:
        # average_power_loss = 2 * sigma^2
        sigma = np.sqrt(average_power_loss / 2)

        # Generate independent Rayleigh fading samples
        rayleigh_samples = sigma * np.random.randn(num_tx,num_rx) + 1j * sigma * np.random.randn(num_tx,num_rx)
        rayleigh_samples = torch.from_numpy(rayleigh_samples).to(device)
        return rayleigh_samples

    def generate_channel_noise(self,signal_tensor:torch.Tensor,device,snr_db=0):
        '''
        Return Noise tensor with same shape and device like signal tensor
        '''
        num_samples = signal_tensor.numel()
        signal_power = torch.sum(signal_tensor**2)/num_samples
        sigma = np.sqrt(signal_power/np.power(10,snr_db/10.0))
        noise = sigma * np.random.randn(num_samples)
        # noise = torch.from_numpy(noise).to(device)
        noise = noise.reshape_as(signal_tensor)
        return noise

    def channel_process(self,w):
        '''
        Apply MIMO channel porcess (precode, over the air, postcode), return added weight

        w: list of all clients' weights
        '''
        device = 'cuda' if self.args.gpu else 'cpu'
        if self.num_tx == 1:    # Support K x SIMO currently
            rayleigh_coefficient = self.generate_channel_gain(len(w),self.num_rx,device)       # num_users(K) x num_rx
            w_avg = copy.deepcopy(w[0])
            if self.args.oac_method == 'none':
                postcode = torch.ones(self.num_rx).to(device)/self.num_rx
            elif self.args.oac_method == 'naive':
                postcode = torch.ones(self.num_rx).to(device)/self.num_rx
            elif self.args.oac_method == 'mimo_oac':
                # Put your postcoding here: (postcode: A)
                h_coefficient = rayleigh_coefficient.cpu().numpy().T    # num_rx x num_users
                csi_G = np.zeros((self.num_rx,self.num_rx),dtype=np.complex64)
                for i in range(len(w)):
                    ui,sigmai,_ = np.linalg.svd(h_coefficient[:,i].reshape(-1,1))
                    csi_G += np.min(sigmai*sigmai)*np.matmul(ui,ui.T.conjugate())
                F_mat,_,_ = np.linalg.svd(csi_G)
                F_mat = F_mat[:,0].conjugate()
                eta = max([np.abs(1/(F_mat.T @ h_coefficient[:,i].reshape(-1,1) @ h_coefficient[:,i].reshape(-1,1).T.conjugate() @ F_mat.conjugate())) for i in range(len(w))])/1
                postcode_np = np.sqrt(eta)*F_mat
                postcode = torch.from_numpy(postcode_np).to(device).flatten()
            elif self.args.oac_method == 'mimo_as':
                h_coefficient = rayleigh_coefficient.cpu().numpy().T    # num_rx x num_users
                h_coefficient_norm = np.sum(h_coefficient,axis=1)
                h_mean = np.mean(h_coefficient_norm)
                mask = h_coefficient_norm>h_mean
                h_coefficient_norm[mask] = 1
                h_coefficient_norm[1-mask] = 0
                postcode_np = h_coefficient_norm
                postcode = torch.from_numpy(postcode_np).to(device).flatten()
            elif self.args.oac_method == 'mimo_eigen':
                h_coefficient = rayleigh_coefficient.cpu().numpy().T    # num_rx x num_users
                h_sum = np.sum(h_coefficient,axis=1).reshape(-1,1)
                uh,sigmah,_ = np.linalg.svd(h_sum)
                postcode_np = uh[:,0]
                postcode = torch.from_numpy(postcode_np).to(device).flatten()
            # postcode = torch.ones(self.num_rx).to(device)

            for i in range(len(w)):  # Each cients' SIMO channel
                # Put your precoding here: (precode: B)
                if self.args.oac_method == 'none':
                    precode = torch.ones(self.num_tx).to(device).flatten()
                elif self.args.oac_method == 'naive':
                    precode = 1/rayleigh_coefficient[i,0].view(-1)
                elif self.args.oac_method == 'mimo_oac' or self.args.oac_method == 'mimo_as' or self.args.oac_method == 'mimo_eigen':
                    precode = (h_coefficient[:,i].reshape(-1,1).T.conjugate() @ postcode_np.conjugate()) * \
                            1/(postcode_np.T @ h_coefficient[:,i].reshape(-1,1) @ h_coefficient[:,i].reshape(-1,1).T.conjugate() @ postcode_np.conjugate())
                    precode = torch.from_numpy(precode.conjugate()).to(device).flatten()
                # yij = hij*gij*x + ni, yi = Σ pj*yij
                for key in w_avg.keys():
                    # attenna 0
                    y_complex = rayleigh_coefficient[i,0] * precode[0] * w[i][key] + self.generate_channel_noise(w[i][key],device,self.args.snr_db)
                    y_post_complex = postcode[0].conj() * y_complex
                    # attenna 1-X
                    for rx in range(1,self.num_rx):
                        y_complex = rayleigh_coefficient[i,rx] * precode[0] * w[i][key] + self.generate_channel_noise(w[i][key],device,self.args.snr_db)
                        y_post_complex += postcode[rx].conj() * y_complex
                    # 存疑，如何处理信号的符号 直接取实部
                    # sign_mask = torch.real(y_post_complex)>0
                    # sign_mask = sign_mask*1.0
                    # sign_mask[sign_mask==1] = 1.0
                    # sign_mask[sign_mask==0] = -1.0
                    # if i==0:
                    #     w_avg[key] = torch.abs(y_post_complex)*sign_mask
                    # else:
                    #     w_avg[key] += torch.abs(y_post_complex)*sign_mask

                    if i==0:
                        w_avg[key] = torch.real(y_post_complex)
                    else:
                        w_avg[key] += torch.real(y_post_complex)

                    if i==(len(w)-1):
                        w_avg[key] = torch.div(w_avg[key], len(w))

        return w_avg

if __name__ == '__main__':
    import argparse
    import torch.nn as nn
    from utils import average_weights
    from tqdm import tqdm
    # 1. 参数两两放到实部和虚部
    # OFDM H对每个子载波选择性衰落 BAA Zhu
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--num_tx', type=int, default=1)
    parser.add_argument('--num_rx', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--snr_db', type=int, default=50)
    parser.add_argument('--oac_method', type=str, default='mimo_eigen')
    args = parser.parse_args()

    def eval_model_mse(w1,w2):
        mseloss = nn.MSELoss()
        val = []
        for key in w1.keys():
            val.append(mseloss(w1[key],w2[key]).numpy())
        return np.mean(np.array(val))

    # 简单测试
    channel = wireless_channel(args)
    model = {'layer1':torch.Tensor([1,2,-1]).to('cpu')}
    model_list = [model]*20
    res = channel.channel_process(model_list)
    print(res)
    print('Model mse:',eval_model_mse(model,res))


    # result_list = []
    # for iter in tqdm(range(10)):
    #     channel = wireless_channel(args)
    #     model_list = []
    #     for client in range(10):
    #         model = {'layer1':torch.Tensor([1,2,-1]).to('cpu'),'layer2':torch.empty(200,10).uniform_(-10,10).to('cpu')}
    #         model_list.append(model)
    #     model_target = average_weights(model_list)
    #     # print(channel.channel_process(model_list))
    #     # print('Model mse:',eval_model_mse(model,channel.channel_process(model_list)))
    #     result_list.append(eval_model_mse(model_target,channel.channel_process(model_list)))
    # print(result_list)
    # print('Avg MSE:',np.mean(np.array(result_list)))
