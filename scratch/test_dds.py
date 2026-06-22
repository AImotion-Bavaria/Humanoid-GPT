import sys
import os
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG

print("Initializing ChannelFactory...")
ChannelFactoryInitialize(0, "enp4s0")

print("Creating ChannelPublisher with actual topic name...")
try:
    pub = ChannelPublisher("rt/lowcmd", LowCmdHG)
    print("ChannelPublisher with actual name created successfully!")
except Exception as e:
    print("Failed to create actual ChannelPublisher:")
    import traceback
    traceback.print_exc()
