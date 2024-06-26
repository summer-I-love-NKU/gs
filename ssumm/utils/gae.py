

def create_gae(args):
    if 0:#os.path.exists(data.result_path) and not args.rerun:
        pass
        # print("GAE Result exists. Loading...")
        # P_onehot = np.load(data.result_path)
    else:
        print("GAE...")
        features=get_label_feature(args)
        input_size = features.shape[1]
        args = args_gcn(args)
        set_seed(args)

        from torch_geometric.nn import GAE
        encoder = GCN(input_size, args.hidden_size, args.gae_out)
        model = GAE(encoder).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        min_loss = 1e9
        patience_num = 0
        patience = 100
        output = None
        best_output=None
        best_str=''
        for epoch in range(args.epochs):
            # t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, args.edge_index)
            loss_train = model.recon_loss(output, args.edge_index)
            loss_train.backward()
            optimizer.step()
            print('Epoch: {:04d}'.format(epoch + 1) + f"  loss:{loss_train}")

            if loss_train < min_loss:
                min_loss = loss_train
                best_output=output
                patience_num = 0
                best_str=f"Best Result:  Epoch {epoch + 1}  min_loss {min_loss}"
            else:
                patience_num += 1
                if patience_num > patience:
                    print('Epoch: {:04d}'.format(epoch + 1) + '  early stop!!!')
                    break
        # res = F.log_softmax(output, dim=1)
        # res = res.max(1)[1].tolist()
        print(best_str)
        best_output = best_output.detach().numpy()
    return best_output

