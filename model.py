def select_model(model_name, args, data_feature, device):
    if model_name == 'SDL':
        from SDL import SDL
        encoder_dim = 128 if args.market_name == 'NASDAQ' else 256
        temporal_embed_dim = 128 if args.market_name == 'NASDAQ' else 256
        config = {
            'input_time': args.input_window,
            'output_time': 1,
            'gcn_dim': 256,  # 256
            'encoder_dim': encoder_dim,  # 128 for NASDAQ, 256 for NYSE
            'temporal_embed_dim': temporal_embed_dim,  # 128 for NASDAQ, 256 for NYSE
            'cheby_k': 2
        }
        model = SDL(config=config, data_feature=data_feature).to(device)

    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return model
