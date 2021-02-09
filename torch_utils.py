import torch


def get_autoencoder(config):
    layer_config = config["layers"]
    encoder = []
    activation = getattr(torch.nn, config["activation"])()

    for i, (inp, otp) in enumerate(zip(layer_config[:-1], layer_config[1:])):
        layer = torch.nn.Linear(inp, otp)
        encoder.append(layer)
        if i != len(layer_config) - 2:
            encoder.append(activation)

    decoder = []
    layer_config = list(reversed(layer_config))
    for i, (inp, otp) in enumerate(zip(layer_config[:-1], layer_config[1:])):
        layer = torch.nn.Linear(inp, otp)
        decoder.append(layer)
        if i != len(layer_config) - 2:
            decoder.append(activation)

    net = torch.nn.Sequential()
    net.add_module("encoder", torch.nn.Sequential(*encoder))
    net.add_module("decoder", torch.nn.Sequential(*decoder))
    return net


def run_autoencoder(data, config):
    autoencoder_net = get_autoencoder(config)
    print(autoencoder_net)
    data_tensor = torch.from_numpy(data).to(dtype=torch.float)
    optimizer = torch.optim.Adam(autoencoder_net.parameters(), lr=1e-03, weight_decay=1e-04)

    loss_fn = torch.nn.MSELoss()
    for epoch in range(config["max_epochs"]):
        autoencoder_net.zero_grad()
        recons_ = autoencoder_net(data_tensor)
        loss_val = loss_fn(recons_, data_tensor)
        loss_val.backward()
        optimizer.step()

    return (
        lambda x: autoencoder_net.encoder(torch.from_numpy(x).to(dtype=torch.float))
        .detach()
        .numpy()
    )
