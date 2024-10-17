import base64
import io

import torch


def stateDictToBase64(stateDict):
    buffer = io.BytesIO()
    torch.save(stateDict, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read())


def base64ToStateDict(b64):
    bytesData = base64.b64decode(b64)
    buffer = io.BytesIO(bytesData)
    return torch.load(buffer)


def stateDictToHex(stateDict):
    buffer = io.BytesIO()
    torch.save(stateDict, buffer)
    buffer.seek(0)
    return bytes.hex(buffer.read())


def hexToStateDict(hex_data):
    try:
        bytes_data = bytes.fromhex(hex_data)
        buffer = io.BytesIO(bytes_data)
        state_dict = torch.load(buffer)  # 加载模型参数
        return state_dict
    except Exception as e:
        print(f"Error in hexToStateDict: {e}")
        return None
