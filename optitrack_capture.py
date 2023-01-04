import optirx as rx

dsock = rx.mkdatasock()
version = (2, 9, 0, 0)  # NatNet version to use
while True:
    data = dsock.recv(rx.MAX_PACKETSIZE)
    packet = rx.unpack(data, version=version)
    if type(packet) is rx.SenderData:
        version = packet.natnet_version
    print(packet)