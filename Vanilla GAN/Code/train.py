import torch
from torch import optim
from torch import  nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ,plotVoxelVisdom
import os
import time
import numpy as np
from utils import ShapeNetDataset, var_or_cuda
from model import _G, _D
from visdom import Visdom 

def train(args):
    #for creating the visdom object
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    viz = Visdom(DEFAULT_HOSTNAME,DEFAULT_PORT, ipv6=False)
    
    hyparam_list = [("model", args.model_name),
                    ("cube", args.cube_len),
                    ("bs", args.batch_size),
                    ("g_lr", args.g_lr),
                    ("d_lr", args.d_lr),
                    ("z", args.z_dis),
                    ("bias", args.bias),
                    ("sl", args.soft_label),]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(log_param)

    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf

        summary_writer = tf.summary.FileWriter(args.output_dir + args.log_dir + log_param)

        def inject_summary(summary_writer, tag, value, step):
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary


    # datset define
    dsets_path = args.input_dir + args.data_dir + "train/"
    print(dsets_path)
    
    x_train = np.load("voxels_3DMNIST_16.npy")
    dataset = x_train.reshape(-1,args.cube_len*args.cube_len*args.cube_len)
    print(dataset.shape)
    dset_loaders = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model define
    D = _D(args)
    G = _G(args)

    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)

    if torch.cuda.is_available():
        print("using cuda")
        D.cuda()
        G.cuda()

    criterion = nn.BCELoss()

    pickle_path = "." + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver)

    for epoch in range(args.n_epochs):
        epoch_start_time = time.time() 
        print("epoch %d started" %(epoch))
        for i, X in enumerate(dset_loaders):

            X = var_or_cuda(X)
            X = X.type(torch.cuda.FloatTensor)
            if X.size()[0] != int(args.batch_size):
                #print("batch_size != {} drop last incompatible batch".format(int(args.batch_size)))
                continue

            Z = generateZ(args)
            real_labels = var_or_cuda(torch.ones(args.batch_size)).view(-1,1,1,1,1)
            fake_labels = var_or_cuda(torch.zeros(args.batch_size)).view(-1,1,1,1,1)

            if args.soft_label:
                real_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0.9, 1.1)).view(-1,1,1,1,1) ####
                #fake_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0, 0.3)).view(-1,1,1,1,1)
                fake_labels = var_or_cuda(torch.zeros(args.batch_size)).view(-1,1,1,1,1) #####
            # ============= Train the discriminator =============#
            d_real = D(X)
            d_real_loss = criterion(d_real, real_labels)


            fake = G(Z)
            d_fake = D(fake)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss


            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))

            #if 1:
            if d_total_acu <= args.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # =============== Train the generator ===============#

            Z = generateZ(args)

            fake = G(Z)
            d_fake = D(fake)
            g_loss = criterion(d_fake, real_labels)

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()
            #######
            #print(fake.shape)
            #print(fake.cpu().data[:8].squeeze().numpy().shape)

        # =============== logging each iteration ===============#
            iteration = str(G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]]['step'])
            #print(type(iteration))
            #iteration = str(i)
            #saving the model and a image each 100 iteration
            if int(iteration) %300 == 0:
                #pickle_save_path = args.output_dir + args.pickle_dir + log_param
                #save_new_pickle(pickle_save_path, iteration, G, G_solver, D, D_solver) 
                samples = fake.cpu().data[:8].squeeze().numpy()
                
                #print(samples.shape)
                for s in range(8):
                    plotVoxelVisdom(samples[s,...], viz, "Iteration:{:.4}".format(iteration))
                
#                 image_path = args.output_dir + args.image_dir + log_param
#                 if not os.path.exists(image_path):
#                     os.makedirs(image_path)

#                 SavePloat_Voxels(samples, image_path, iteration)
        # =============== each epoch save model or save image ===============#
            print('Iter-{}; , D_loss : {:.4}, G_loss : {:.4}, D_acu : {:.4}, D_lr : {:.4}'.format(iteration, d_loss.item(), g_loss.item(), d_total_acu.item(), D_solver.state_dict()['param_groups'][0]["lr"]))
        
        epoch_end_time = time.time()
        
        if (epoch + 1) % args.image_save_step == 0:

            samples = fake.cpu().data[:8].squeeze().numpy()

            image_path = args.output_dir + args.image_dir + log_param
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            SavePloat_Voxels(samples, image_path, iteration)

        if (epoch + 1) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, iteration, G, G_solver, D, D_solver)

        print("epoch time", (epoch_end_time-epoch_start_time)/60)
        print("epoch %d ended" %(epoch))
        print("################################################")
