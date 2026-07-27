from frontierguard.io import write_json
from frontierguard.utils.environment import audit_environment


if __name__ == "__main__":
    value = audit_environment()
    write_json("environment.json", value)
    print("wrote environment.json")
