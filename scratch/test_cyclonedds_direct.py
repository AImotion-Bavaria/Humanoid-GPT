import sys
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

print("Creating DomainParticipant...")
try:
    participant = DomainParticipant()
    print("DomainParticipant created successfully!")
except Exception as e:
    print(f"Failed to create DomainParticipant: {e}")
    sys.exit(1)

print("Creating Topic...")
try:
    topic = Topic(participant, "rt/lowcmd", LowCmd_)
    print("Topic created successfully!")
except Exception as e:
    print("Failed to create Topic:")
    import traceback
    traceback.print_exc()
