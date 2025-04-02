def register(address, dataSize):
    global globalState
    if address in globalState['uid']:
        return False  # 节点已存在

    uid = globalState['n_d']  # 分配一个新的 UID
    globalState['uid'][address] = uid
    globalState['n_d'] += 1  # 更新节点计数
    
    # 确保所有评分列表被正确初始化
    for i in range(4):  # s有4个子列表
        while len(globalState['s'][i]) < globalState['n_d']:
            default_value = 0.0
            if i == 1:  # 对于得分列表使用0.5作为默认值
                default_value = 0.5
            elif i == 2:  # 对于Tau值使用1作为默认值
                default_value = 1.0
            globalState['s'][i].append(default_value)
    
    globalState['scores'].append(0)
    globalState['t'].append(time.time())
    globalState['s'][0][uid] = dataSize  # 设置数据大小
    
    return True
import base64
import io
import requests
import torch
import copy # Import copy for deepcopy
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import time

from models.Nets import CNN
# Import fed_avg_aggregate if needed, but we decided against FedAvg for now
from models.Fed import getDis, getAlpha, getTauI, normalization # BAFL functions
from models.Fed import getAlpha_iewm, getTauI_iewm, normalization_iewm # IEWM functions
from utils.options import args_server_parser
from utils.models import hexToStateDict, stateDictToHex

# 解析服务器参数
args = args_server_parser() # args is now available globally in this module

# 初始化全局模型
def initNetGlob():
    if args.model == 'cnn':
        net_glob = CNN(num_classes=args.num_classes, num_channels=args.num_channels)
    else:
        exit('Error: unrecognized model')
    return net_glob

# 初始化 Flask 应用和 SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# 初始化全局模型
net_glob = initNetGlob()

# 全局状态，用于管理多个节点的状态
globalState = {
    'n_d': 0,  # 节点数
    's': [
        [], [], [], []
    ],  # 存储分数和距离等信息
    't': [],  # 时间戳
    'scores': [],  # 各个节点的得分
    'uid': {}  # 节点的 UID 映射
}

# 模型聚合的结果
weights = []

def merge(uid, address, data):
    try:
        print(f"\nMerging local model from node: {address}")
        print(f"Current state - UID: {uid}")
        print(f"Data state: n_d={data['n_d']}, t0={data['t0']}")
        print(f"S matrix shape: {[len(s) for s in data['s']]}")
        
        # 验证输入数据
        if 'uid' not in data or 'n_d' not in data or 's' not in data:
            print("Missing required data fields")
            return None
            
        alpha = getAlpha(1, int(time.time()), data['t0'], 0.003, 1, data['uid'], data['n_d'], data['s'])
        print(f"Calculated alpha: {alpha}")
        
        if alpha == 0:
            print("Alpha is 0, skipping merge")
            return None
            
        # 模型更新
        try:
            localStateDict = data['local_state_dict']
            globStateDict = data['global_state_dict']
            
            for k in globStateDict.keys():
                globStateDict[k] = (1 - alpha) * globStateDict[k] + alpha * localStateDict[k]
                
            stateDictHex = stateDictToHex(globStateDict)
            s = normalization(data['s'])
            score = getTauI(uid, data['n_d'], s)
            
            print(f"Merge completed. Score: {score}")
            
            return {
                'model_state_hex': stateDictHex,
                'score': score,
                'address': address,
                'cur_global_state_dict': globStateDict
            }
        except Exception as e:
            print(f"Error during model update: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error in merge: {str(e)}")
        return None

# 接收并合并节点的本地模型
@app.route('/newLocalModel/<address>', methods=['POST'])
def newLocalModel(address):
    data = request.json
    localStateDict = hexToStateDict(data['local_model_hex'])
    
    global net_glob
    globalStateDict = net_glob.state_dict()  # 获取当前全局模型的状态字典
    uid = globalState['uid'][address]  # 获取节点的 UID

    if args.aggregate == 'bafl':
        # --- Existing BAFL Logic ---
        globalState['s'][3][uid] = getDis(globalStateDict, localStateDict) # Update distance for BAFL

        res = merge(uid, address, { # merge uses getAlpha which uses entropy weights
            "local_state_dict": localStateDict,
            "global_state_dict": globalStateDict,
            "s": globalState['s'],
            "n_d": globalState['n_d'],
            "uid": uid,
            "t0": globalState['t'][uid]
        })

        if res:
            # Update global state (time, score) specific to BAFL
            globalState['t'][uid] = time.time()
            globalState['s'][1][uid] = float(res['score'])
            net_glob.load_state_dict(hexToStateDict(res['model_state_hex'])) # Update global model
            print(f"Model from {address} successfully merged into global model using BAFL.")
        else:
             print(f"BAFL merge returned None for node {address}. Global model not updated.")
             print(f"BAFL merge returned None for node {address}. Global model not updated.")
             # Decide if we should still update the timestamp
             # globalState['t'][uid] = time.time() # Update time even if merge fails?

    elif args.aggregate == 'iewm':
        # --- Information Entropy Weight Method (IEWM) Logic ---
        globalState['s'][3][uid] = getDis(globalStateDict, localStateDict) # Update distance metric

        # Parameters for getAlpha_iewm (using similar defaults as BAFL's merge for now)
        kexi = 1.0
        theta = 0.003
        R0 = 1.0
        current_time = time.time()
        t0 = globalState['t'][uid]
        N_D = globalState['n_d']
        s_metrics = globalState['s'] # Use the same metrics as BAFL for now

        # Calculate alpha using IEWM functions
        alpha_iewm = getAlpha_iewm(kexi, current_time, t0, theta, R0, uid, N_D, s_metrics)
        print(f"Calculated IEWM alpha for node {address}: {alpha_iewm}")

        if alpha_iewm > 0: # Proceed only if alpha is positive
            updated_global_dict_iewm = copy.deepcopy(globalStateDict)
            try:
                for k in updated_global_dict_iewm.keys():
                    if k in localStateDict:
                        updated_global_dict_iewm[k] = (1.0 - alpha_iewm) * updated_global_dict_iewm[k] + alpha_iewm * localStateDict[k]
                    else:
                        print(f"Warning (IEWM): Key '{k}' not found in local model from {address}. Skipping update for this layer.")

                net_glob.load_state_dict(updated_global_dict_iewm) # Update global model

                # Update timestamp
                globalState['t'][uid] = current_time

                # Optionally calculate and update score based on IEWM's TauI
                try:
                    # Use the normalized metrics as used within getAlpha_iewm
                    s_normalized_iewm = normalization_iewm(s_metrics)
                    if s_normalized_iewm: # Check if normalization was successful
                        score_iewm = getTauI_iewm(uid, N_D, s_normalized_iewm)
                        globalState['s'][1][uid] = float(score_iewm)
                        print(f"IEWM Score calculated for node {address}: {score_iewm}")
                    else:
                         print(f"Warning (IEWM): Normalization failed, cannot calculate score for node {address}.")
                         globalState['s'][1][uid] = 0.5 # Reset to default?
                except Exception as e:
                    print(f"Error calculating IEWM score for node {address}: {e}")
                    globalState['s'][1][uid] = 0.5 # Reset to default on error

                print(f"Model from {address} successfully merged into global model using IEWM (alpha={alpha_iewm}).")

            except Exception as e:
                print(f"Error during IEWM aggregation for node {address}: {e}")
                # Don't return '0' here as it might break client expectation of '1'
                # Just log the error and the global model won't be updated by this client
        else:
            print(f"IEWM alpha is 0 or less for node {address}, skipping merge.")
            # Optionally update timestamp even if merge is skipped?
            # globalState['t'][uid] = current_time

    elif args.aggregate == 'fedasync':
        # --- Simple FedAsync Logic ---
        # Use a fixed averaging factor (alpha)
        alpha_async = 0.1 # Example fixed alpha for simple async averaging
        updated_global_dict = copy.deepcopy(globalStateDict)
        try:
            for k in updated_global_dict.keys():
                 if k in localStateDict:
                      # Ensure tensors are on the same device and dtype if necessary
                      # Assuming they are compatible based on previous code structure
                      updated_global_dict[k] = (1.0 - alpha_async) * updated_global_dict[k] + alpha_async * localStateDict[k]
                 else:
                     print(f"Warning: Key '{k}' not found in local model from {address}. Skipping update for this layer.")

            net_glob.load_state_dict(updated_global_dict) # Update global model
            # Update timestamp, score is not calculated in simple FedAsync
            globalState['t'][uid] = time.time()
            # Optionally reset or ignore score: globalState['s'][1][uid] = 0.5 # Reset to default?
            print(f"Model from {address} successfully merged into global model using FedAsync (alpha={alpha_async}).")
        except Exception as e:
            print(f"Error during FedAsync aggregation for node {address}: {e}")
            return '0', 500 # Indicate server error

    else:
        print(f"Error: Unknown aggregation method '{args.aggregate}'")
        return '0', 500 # Internal server error

    return '1' # Indicate success

# 返回或上传全局模型
@app.route('/newGlobalModel', methods=['GET', 'POST'])
def newGlobalModel():
    global net_glob
    if request.method == 'POST':
        # 从客户端接收到的全局模型
        data = request.json
        globalModelStateDict = hexToStateDict(data['global_model_hex'])
        net_glob.load_state_dict(globalModelStateDict)
        return '1'
    elif request.method == 'GET':
        # 返回当前全局模型的十六进制状态
        if net_glob is None:
            print("No global model available.")
            return jsonify({"error": "No global model available"}), 500
        
        state_dict_hex = stateDictToHex(net_glob.state_dict())
        print(f"Returning global model state: {state_dict_hex[:50]}...")  # 打印部分十六进制以确认返回值
        return jsonify({'global_model_hex': state_dict_hex})

# 注册新的节点
def register(address, dataSize):
    global globalState
    if address in globalState['uid']:
        return False  # 节点已存在

    uid = globalState['n_d']  # 分配一个新的 UID
    globalState['uid'][address] = uid
    globalState['n_d'] += 1  # 更新节点计数
    globalState['scores'].append(0)
    globalState['t'].append(time.time())  # 保存当前时间戳
    globalState['s'][0].append(dataSize)  # 记录数据大小
    globalState['s'][1].append(0.5)  # 默认得分
    globalState['s'][2].append(1)  # 默认 Tau 值
    globalState['s'][3].append(0)  # 默认距离
    return True

# 处理节点的注册请求
@app.route('/register', methods=['POST'])
def handleRegister():
    data = request.json
    address = data['address']
    dataSize = int(data['data_size'])
    if register(address, dataSize):
        print(f"Node {address} registered successfully with data size {dataSize}.")
        return '1'
    else:
        print(f"Node {address} already registered.")
        return '0'

# 返回全局模型的元信息 (如模型类型和输入输出结构)
@app.route("/getTrainInfo", methods=['GET'])
def getModelType():
    return {
        'model_name': args.model,
        'num_classes': args.num_classes,
        'num_channels': args.num_channels
    }

# 启动服务器
if __name__ == '__main__':
    print(f"Starting server on port {args.port}...")
    socketio.run(app, port=args.port, allow_unsafe_werkzeug=True)
