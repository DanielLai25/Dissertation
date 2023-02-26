"""AAGNN"""
AAGNN_model = AAGNN(feature_size=features_emb.shape[1],
                  hidden_size=args.AAGNN_hidden,
                  dropout_ratio=args.AAGNN_dropout,
                  Graph_networkx=G)

AAGNN_optimizer = optim.Adam(AAGNN_model.parameters(), lr=args.AAGNN_lr, weight_decay=args.AAGNN_weight_decay)
AAGNN_model.to(device)

features_emb = features_emb.to(device)

AAGNN_model.eval()
polluted_train_emb = AAGNN_model(features_emb, adj_dense, degree_mat).detach().cpu().numpy()
polluted_train_embed = torch.FloatTensor(polluted_train_emb)
center = torch.mean(polluted_train_embed, 0).to(device)

AAGNN_model.to(device)

if not os.path.exists(AAGNN_para_name):
    print("no AAGNN parameter exists")

    early_stopping = EarlyStopping(patience=args.patient, verbose=True, path=AAGNN_para_name)

    t_total = time.time()
    for epoch in range(args.AAGNN_epochs):
        t = time.time()
        AAGNN_model.train()
        AAGNN_optimizer.zero_grad()
        output = AAGNN_model(features_emb, adj_dense, degree_mat)
        GDN_loss_train = objecttive_loss_valid(output[train_final_indx], center)
        # GDN_loss_train = objecttive_loss_valid(output, center)
        GDN_loss_train.backward(retain_graph=True)
        AAGNN_optimizer.step()

        # evaluate in val set

        AAGNN_model.eval()
        output_val = AAGNN_model(features_emb, adj_dense, degree_mat)
        loss_val = objecttive_loss_valid(output_val[valid_final_indx], center)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.30f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        early_stopping(loss_val, AAGNN_model)  # validation loss
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    torch.save(AAGNN_model.state_dict(), AAGNN_para_name)
    print("AAGNN parameter saved")

else:
    print("AAGNN parameter loaded")
    AAGNN_model.load_state_dict(torch.load(AAGNN_para_name))



"""autoencoder model"""
autoencoder_model = Autoencoder(input_size = features_pre_train.shape[1],
                                first_layer_size = args.auto_1st_layer,
                                second_layer_size = args.auto_2nd_layer,
                                dropout_ratio = args.auto_dropout)

autoencoder_optimizer = optim.Adam(autoencoder_model.parameters(), lr=args.auto_lr,\
                                   betas=(args.auto_beta1, args.auto_beta2),eps=1e-8)
autoencoder_model.to(device)
features_pre_train = features_pre_train.to(device)

if not os.path.exists(auto_para_name):
    print("no auto parameter exists")

    early_stopping = EarlyStopping(patience=args.patient, verbose=True, path=auto_para_name)
    for epoch in range(args.auto_epochs):
        t = time.time()
        autoencoder_model.train()
        autoencoder_optimizer.zero_grad()
        _, output = autoencoder_model(features_pre_train)
        emb_loss_train = nn.MSELoss()(output[train_final_indx], features_pre_train[train_final_indx])
        # emb_loss_train = nn.MSELoss()(output, features_nor)
        emb_loss_train.backward()
        autoencoder_optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.30f}'.format(emb_loss_train.item()),
              'time: {:.4f}s'.format(time.time() - t))

        # autoencoder_model.eval()
        # _, output_val = autoencoder_model(features_pre_train)
        # loss_val = nn.MSELoss()(output_val[valid_final_indx], features_pre_train[valid_final_indx])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.30f}'.format(loss_val.item()),
        #       'time: {:.4f}s'.format(time.time() - t))

    torch.save(autoencoder_model.state_dict(), auto_para_name)
    print("auto parameter saved")

else:
    print("auto parameter loaded")
    autoencoder_model.load_state_dict(torch.load(auto_para_name))

