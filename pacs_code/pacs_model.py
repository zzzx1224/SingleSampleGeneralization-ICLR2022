import os
import sys
import time
import math
import random
import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import models
import numpy as np 
import pdb
import torch.nn.functional as f
# from main import args

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)
        init.xavier_uniform(m.bias)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        init.constant(m.bias, 0)

class classifier_generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(classifier_generator, self).__init__()
        self.shared_net = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU()
                        )
        self.shared_mu = nn.Linear(hidden_size, output_size)
        self.shared_sigma = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z = self.shared_net(x)
        return self.shared_mu(z), self.shared_sigma(z)

class net0(nn.Module):
    def __init__(self, num_class, num_source_domains, backbone, dinit, agg_model='concat', agg_method='ensemble', dsgrad=True, 
                zdlayers=4, dom_sta='both', prior_test=False, hierar=0, adap_samples=1, multi_train_samp=False, sharemodel=True, mc_times=10, with_bias=True):
        super(net0, self).__init__()
        self.num_class = num_class
        self.with_bias = with_bias
        self.num_d = num_source_domains
        self.mc_times = mc_times
        self.zdlayers = zdlayers
        self.dinit = dinit
        self.agg_model = agg_model
        self.agg_method = agg_method
        self.prior_test = prior_test
        self.sharemodel = sharemodel
        self.dom_sta = dom_sta
        self.hierar = hierar
        self.dsgrad = dsgrad
        self.test_adap_samples = adap_samples
        if multi_train_samp:
            self.train_adap_samples = adap_samples
        else:
            self.train_adap_samples = '1'

        # backbone
        if backbone=='res18':
            resnet = resnet18
            self.feature_dim = 512

            if zdlayers==4:
                len_zd = (resnet.layer1[-1].conv2.out_channels + resnet.layer2[-1].conv2.out_channels + resnet.layer3[-1].conv2.out_channels + resnet.layer4[-1].conv2.out_channels) * (2**int((dom_sta=='both')))
            elif zdlayers==3:
                len_zd = (resnet.layer1[-1].conv2.out_channels + resnet.layer2[-1].conv2.out_channels + resnet.layer3[-1].conv2.out_channels) * (2**int((dom_sta=='both')))
            elif zdlayers==1:
                len_zd = (resnet.layer4[-1].conv2.out_channels) * (2**int((dom_sta=='both')))

        elif backbone=='res50':
            resnet = resnet50
            self.feature_dim = 2048

            if zdlayers==4:
                len_zd = (resnet.layer1[-1].conv3.out_channels + resnet.layer2[-1].conv3.out_channels + resnet.layer3[-1].conv3.out_channels + resnet.layer4[-1].conv3.out_channels) * (2**int((dom_sta=='both')))
            elif zdlayers==3:
                len_zd = (resnet.layer1[-1].conv3.out_channels + resnet.layer2[-1].conv3.out_channels + resnet.layer3[-1].conv3.out_channels) * (2**int((dom_sta=='both')))
            elif zdlayers==1:
                len_zd = (resnet.layer4[-1].conv3.out_channels) * (2**int((dom_sta=='both')))

        self.layer0 = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool
                    )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = resnet.avgpool

        # aggregation
        # self.domain_layer = nn.Linear(len_zd, self.feature_dim)
        self.domain_layer = nn.Sequential(
                            nn.Linear(len_zd, self.feature_dim),
                            nn.ReLU(),
                            nn.Linear(self.feature_dim, self.feature_dim),
                            # nn.LayerNorm(self.feature_dim)
                            )
        # domain_layer = nn.Linear(len_zd+self.feature_dim, self.feature_dim)
        self.agg_layer = nn.Sequential(
                        nn.Linear(self.feature_dim*2, self.feature_dim),
                        nn.ReLU(),
                        nn.Linear(self.feature_dim, self.feature_dim),
                        # nn.LayerNorm(self.feature_dim)
                        )
        # self.agg_layer = nn.Sequential(
        #                 domain_layer,
        #                 nn.ReLU(),
        #                 nn.Linear(self.feature_dim, self.feature_dim)
        #                 )
        self.shared_net = classifier_generator(self.feature_dim, self.feature_dim, self.feature_dim)
        if hierar==2:
            self.shared_net1 = classifier_generator(self.feature_dim, self.feature_dim, self.feature_dim)

        # context info
        self.context = []

    def forward(self, x, ctx, domain_id=0, num_ctx=20, ifsample=True):
        # pdb.set_trace()
        self.num_ctx = num_ctx
        self.domain_id = domain_id

        z0 = self.layer0(x)
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        z = self.pool(z4)

        ctx = self.layer0(ctx)
        ctx = self.layer1(ctx)
        ctx = self.layer2(ctx)
        ctx = self.layer3(ctx)
        ctx = self.layer4(ctx)
        ctx = self.pool(ctx)
        # pdb.set_trace()

        ctx = ctx.view(3, self.num_class, num_ctx, self.feature_dim)

        ctx = ctx.mean(2)

        if not self.dsgrad:
            ctx = ctx.detach()

        # pdb.set_trace()
        z1 = z1.mean(-1).mean(-1)
        z2 = z2.mean(-1).mean(-1)
        z3 = z3.mean(-1).mean(-1)
        z4 = z4.mean(-1).mean(-1)

        if self.zdlayers==4:
            zd = torch.cat([z1, z2, z3, z4], 1)
        elif self.zdlayers==3:
            zd = torch.cat([z1, z2, z3], 1)
        elif self.zdlayers==1:
            zd = z4
        self.z = z.squeeze()
        # print(self.z)
        zd = zd.squeeze()
        #zd = torch.zeros(zd.size()).cuda()
        # pdb.set_trace()
        # yc, common_kl = self.com_classifier(self.z, self.mc_times, ifsample)
        
        if self.training:
            self.ws, pws, KLD, hkld = self.adaptive_weight_training(zd, self.z, ctx, domain_id) # 128*10*14*512
            # pdb.set_trace()
            pws = pws.view(self.feature_dim, self.mc_times*self.num_class) # 512*(10*7)
            yps = torch.mm(self.z, pws) # 128*512  512*(10*7) => 128*(10*7)
            yps = yps.view(yps.size()[0], self.mc_times, self.num_class)
            # pdb.set_trace()
        else:
            self.ws, KLD, hkld = self.target_adaptive_weight(zd, self.z, ctx, ifsample)
                # pdb.set_trace()
            yps = 0 
            hkld = 0

        self.ws = self.ws.view(z.size()[0], -1, self.feature_dim) # 128*(10*2*7)*512
        ys = torch.bmm(self.ws, self.z.unsqueeze(-1)) # 128*(10*2*7)*512   128*512*1 => 128*(10*2*7)*1
        ys = ys.squeeze() # 128*(10*2*7)

        ys = ys.view(ys.size()[0], self.mc_times**int(ifsample), -1) #128*10*2*7  common adapt in test
        # ys = ys.view(ys.size()[0], self.mc_times, -1) #128*10*2*7   test no adapt

        # return yc, ys, yps, common_kl, KLD
        return ys, yps, KLD, hkld

    def adaptive_weight_training(self, zd, zt, c, domain_id, add_kl=False):
        # pdb.set_trace()
        if self.dinit=='rt':
            self.context = []
            for i in range(c.size()[0]):
                self.context.append(c[i])

        current_target_domain = self.context.pop(domain_id)

        num_sd = 1
        current_source_domains = torch.cat(self.context, 0).view(len(self.context), self.num_class, self.feature_dim).mean(0) # 7*512

        p_mu, p_v = self.shared_net(current_target_domain) # 7*512
            
        self.context.insert(domain_id, current_target_domain)

        # posterior
        if self.zdlayers==1:
            zd = zd
        else:
            zd = self.domain_layer(zd)  # 128*512

        if self.train_adap_samples == 'all':
            # pdb.set_trace()
            bs = zd.size()[0]
            zd = zd.mean(0, True).repeat(bs, 1)
        elif self.train_adap_samples != '1':
            # pdb.set_trace()
            # zd = zd.view(-1, int(self.train_adap_samples), self.feature_dim).mean(1, True)
            # zd = zd.repeat(1, int(self.train_adap_samples), 1).view(-1, self.feature_dim)
            multi_feat = zd[:int(self.train_adap_samples)-1]
            # pdb.set_trace()
            multi_feat = multi_feat.sum(0, True)
            multi_feat = multi_feat.repeat(zd.size()[0], 1)
            zd = (zd + multi_feat) / int(self.train_adap_samples)

        if self.hierar==0:
            ws = current_source_domains
            # H_kld = torch.zeros(self.num_class, self.mc_times, self.feature_dim).cuda()
            H_kld = torch.zeros(1).cuda()
        elif self.hierar==2:
            ws_mu, ws_v = self.shared_net1(current_source_domains) # (2*7)*512
            # pdb.set_trace()
            ws_sigma = f.softplus(ws_v, beta=1, threshold=20)
            ws_mu_samp = ws_mu.unsqueeze(1).repeat(1, self.mc_times, 1) # (2*7)*10*512
            ws_sigma_samp = ws_sigma.unsqueeze(1).repeat(1, self.mc_times, 1) # (2*7)*10*512
            # pdb.set_trace()
            eps_ws = ws_mu_samp.new(ws_mu_samp.size()).normal_()
            ws = ws_mu_samp + 1 * ws_sigma_samp * eps_ws # (2*7)*10*512
            # ws = ws_mu_samp

            H_kld = torch.zeros(1).cuda()

        # pdb.set_trace()
        zd = zd.view(zd.size()[0], 1, 1, zd.size()[1]).repeat(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), 1) # 128*14*10*512
        if self.agg_method!='att':
            Ds = ws.view(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).repeat(zd.size()[0], 1, 1, 1) # 128*14*10*512
        else:
            Ds = ws.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim)
        q = self.agg_layer(torch.cat([zd, Ds], -1).view(zd.size()[0]*zd.size()[1]*zd.size()[2], zd.size()[-1]+Ds.size()[-1]))
        if self.hierar==3:
            q_mu, q_v = self.shared_net2(q) # (128*14*10)*512
        else:
            q_mu, q_v = self.shared_net(q) # (128*14*10)*512
        q_mu = q_mu.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).mean(2) # 128*14*10*512
        q_v = q_v.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).mean(2) # 128*14*10*512

        # KL divergence
        p_sigma = f.softplus(p_v, beta=1, threshold=20)
        q_sigma = f.softplus(q_v, beta=1, threshold=20)

        p_mu_re = p_mu.view(1, 1, self.num_class, self.feature_dim).repeat(zd.size()[0], num_sd, 1, 1) #128*2*7*512
        p_sigma_re = p_sigma.view(1, 1, self.num_class, self.feature_dim).repeat(zd.size()[0], num_sd, 1, 1)
        p_mu_re = p_mu_re.view(zd.size()[0], num_sd*self.num_class, self.feature_dim) #128*(2*7)*512
        p_sigma_re = p_sigma_re.view(zd.size()[0], num_sd*self.num_class, self.feature_dim)

        KLD = self.kl_divergence(q_mu, q_sigma, p_mu_re, p_sigma_re)
        KLD = KLD.sum()

        # sampling
        q_mu_samp = q_mu.unsqueeze(1).repeat(1, self.mc_times, 1, 1) # 128*10*(2*7)*512
        q_sigma_samp = q_sigma.unsqueeze(1).repeat(1, self.mc_times, 1, 1) # 128*10*(2*7)*512
        eps_q = q_mu_samp.new(q_mu_samp.size()).normal_()
        qw = q_mu_samp + 1 * q_sigma_samp * eps_q

        # pdb.set_trace()
        p_mu_samp = p_mu.transpose(0,1).unsqueeze(1).repeat(1, self.mc_times, 1) # 512*10*7
        p_sigma_samp = p_sigma.transpose(1,0).unsqueeze(1).repeat(1, self.mc_times, 1) # 512*10*7
        eps_p = p_mu_samp.new(p_mu_samp.size()).normal_()
        pw = p_mu_samp + 1 * p_sigma_samp * eps_p

        print(q_mu.mean(), q_sigma.mean())
        print(p_mu.mean(), p_sigma.mean())

        return qw, pw, KLD, H_kld

    def target_adaptive_weight(self, zd, zt, c, ifsample):
        # # pdb.set_trace()
        # if self.dinit=='rt':
        #     self.context = []
        #     for i in range(c.size()[0]):
        #         self.context.append(c[i])

        # pdb.set_trace()
        if self.domain_id < 3:
            current_target_domain = self.context.pop(self.domain_id)

        # pdb.set_trace()

        num_sd = 1
        current_source_domains = torch.cat(self.context, 0).view(len(self.context), self.num_class, self.feature_dim).mean(0) # 7*512

        if self.domain_id < 3:
            self.context.insert(self.domain_id, current_target_domain)

        # posterior
        if self.hierar==0:
            ws = current_source_domains
        elif self.hierar==2:
            ws_mu, ws_v = self.shared_net1(current_source_domains) # (3*7)*512
            # pdb.set_trace()
            ws_sigma = f.softplus(ws_v, beta=1, threshold=20)
            ws_mu_samp = ws_mu.unsqueeze(1).repeat(1, self.mc_times, 1) # (3*7)*10*512
            ws_sigma_samp = ws_sigma.unsqueeze(1).repeat(1, self.mc_times, 1) # (3*7)*10*512
            eps_ws = ws_mu_samp.new(ws_mu_samp.size()).normal_()
            # ws = ws_mu_samp + 1 * ws_sigma_samp * eps_ws # (3*7)*10*512
            ws = ws_mu_samp

        zd = self.domain_layer(zd)  # 128*512

        if self.test_adap_samples == 'all':
            # pdb.set_trace()
            bs = zd.size()[0]
            zd = zd.mean(0, True).repeat(bs, 1)
        elif self.test_adap_samples != '1':
            # pdb.set_trace()
            # zd = zd.view(-1, int(self.test_adap_samples), self.feature_dim).mean(1, True)
            # zd = zd.repeat(1, int(self.test_adap_samples), 1).view(-1, self.feature_dim)
            multi_feat = zd[:int(self.test_adap_samples)-1]
            # pdb.set_trace()
            multi_feat = multi_feat.sum(0, True)
            multi_feat = multi_feat.repeat(zd.size()[0], 1)
            zd = (zd + multi_feat) / int(self.test_adap_samples)

        # pdb.set_trace()
        zd = zd.view(zd.size()[0], 1, 1, zd.size()[1]).repeat(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), 1) # 128*14*10*512
        if self.agg_method!='att':
            Ds = ws.view(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).repeat(zd.size()[0], 1, 1, 1) # 128*14*10*512
        else:
            Ds = ws.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim)
        # zd = zd.view(zd.size()[0], 1, 1, zd.size()[1]).repeat(1,current_source_domains.size()[0], self.mc_times**int(self.hierar!=0), 1) # 128*21*10*512
        # Ds = ws.view(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).repeat(zd.size()[0], 1, 1, 1) # 128*21*10*512
        q = self.agg_layer(torch.cat([zd, Ds], -1).view(zd.size()[0]*zd.size()[1]*zd.size()[2], zd.size()[-1]+Ds.size()[-1]))
        if self.hierar==3:
            q_mu, q_v = self.shared_net2(q) # (128*21*10)*512
        else:
            q_mu, q_v = self.shared_net(q) # 128*21*10*512
        q_mu = q_mu.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).mean(2)
        q_v = q_v.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).mean(2)
        q_sigma = f.softplus(q_v, beta=1, threshold=20)

        KLD = 0
        H_kld = 0

        # sampling
        if ifsample:
            q_mu_samp = q_mu.unsqueeze(2).repeat(1, 1, self.mc_times, 1)
            q_sigma_samp = q_sigma.unsqueeze(2).repeat(1, 1, self.mc_times, 1)
            eps_q = q_mu_samp.new(q_mu_samp.size()).normal_()
            qw = q_mu_samp + 1 * q_sigma_samp * eps_q
        else:
            qw = q_mu

        return qw, KLD, H_kld

    def kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):

        var_q = sigma_q**2 + 1e-6
        var_p = sigma_p**2 + 1e-6

        component1 = torch.log(var_p) - torch.log(var_q)
        component2 = var_q / var_p
        component3 = (mu_p - mu_q).pow(2)/ var_p

        KLD = 0.5 * torch.sum((component1 -1 +component2 +component3),1)
        return KLD