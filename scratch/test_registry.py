import cyclonedds.idl as idl
print("Importing unitree_sdk2py...")
import unitree_sdk2py
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd1
from unitree_sdk2py.idl.unitree_hg.msg.dds_._LowCmd_ import LowCmd_ as LowCmd2

print("Low1 is Low2:", LowCmd1 is LowCmd2)
print("Low1 id:", id(LowCmd1))
print("Low2 id:", id(LowCmd2))

# Let's inspect the registry in cyclonedds
import cyclonedds.idl.types as types
print("Registered types:")
for typename, cls in getattr(idl, "_registry", {}).items():
    print(f"  {typename}: {cls} (id={id(cls)})")

# In cyclonedds 0.10.x, where is the registry stored?
# Let's check internal modules
from cyclonedds import internal
print("Internal stuff:")
for attr in dir(internal):
    if "registry" in attr.lower() or "type" in attr.lower():
        print(" ", attr)
