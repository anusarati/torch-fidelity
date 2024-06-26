def fix_channels(conv, in_channels):
    """Creates new convolutional layer with correct number of input channels"""
    new = type(conv)(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        bias=conv.bias != None,
    )
    least_channels = min(conv.in_channels, new.in_channels)
    new.weight.data[:, :least_channels] = conv.weight.data[:, :least_channels]
    if conv.bias != None:
        new.bias.data = conv.bias.data
    return new
