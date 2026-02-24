import numpy as np
import re
import torch
from torch.autograd import Variable
import time
from chemprop.data.utils import get_data_from_smiles
from chemprop.features import mol2graph
"""基础工具函数"""
def get_torch_device():#返回当前可用的设备（GPU 或 CPU）
    """
    Getter for an available pyTorch device.
    :return: CUDA-capable GPU if available, CPU otherwise
    """
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
def create_variable(tens):#将张量转换为 torch.Variable，并转移到可用设备
    # Do cuda() before wrapping with variable
    return Variable(torch.tensor(tens).to(get_torch_device()))
def replace_halogen(string):#将SMILES中的'Br'→'R','Cl'→'L'替换，以便统一字符
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string
# Create necessary variables, lengths, and target

"""SMILES 序列处理"""
def make_variables(lines, properties,letters):#用于构造序列输入（SMILES → 向量序列）及其标签
    sequence_and_length = [line2voc_arr(line,letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, properties)

def make_variables_s(lines,properties,mpnargs):#将 SMILES 转换为图结构用于图神经网络的输入
    return  mol2graph(lines,mpnargs),create_variable(properties)


def make_variables_seq(lines,letters):#只用于 SMILES 序列输入的向量化和 padding
    sequence_and_length = [line2voc_arr(line,letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences_seq(vectorized_seqs, seq_lengths)
def line2voc_arr(line,letters):#将 SMILES 字符串转化为对应的字符索引数组
    arr = []
    regex = '(\[[^\[\]]{1,10}\])'
    line = replace_halogen(line)
    char_list = re.split(regex, line)
    for li, char in enumerate(char_list):
        if char.startswith('['):
               arr.append(letterToIndex(char,letters)) 
        else:
            chars = [unit for unit in char]

            for i, unit in enumerate(chars):
                arr.append(letterToIndex(unit,letters))
    return arr, len(arr)
def letterToIndex(letter,smiles_letters):#根据字符查找其在 letters 列表中的索引
    return smiles_letters.index(letter)
# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths, properties):#将不等长的 SMILES 序列补齐为等长，并返回输入张量和目标张量
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Also sort the target (countries) in the same order
    target = properties.double()
    if len(properties):
        target = target[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor),create_variable(seq_lengths),create_variable(target)
def pad_sequences_seq(vectorized_seqs, seq_lengths):#
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), create_variable(seq_lengths)

def construct_vocabulary(smiles_list,fname):#
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = ds.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open(fname, 'w') as f:
        f.write('<pad>' + "\n")
        for char in add_chars:
            f.write(char + "\n")
    return add_chars
def readLinesStrip(lines):#
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines

"""蛋白质序列与接触图处理"""
def getProteinSeq(path,contactMapName):#读取蛋白质序列（不含接触图）
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    return seq
def getProtein(path,contactMapName,contactMap = True):#读取蛋白质序列和接触图（可选）
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    if(contactMap):
        contactMap = []
        for i in range(2,len(proteins)):
            contactMap.append(proteins[i])
        return seq,contactMap
    else:
        return seq

"""数据集加载辅助函数"""
def getTrainDataSet(trainFoldPath):#读取训练集 SMILES-序列-标签数据
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    trainDataSet = [cpi.strip().split() for cpi in trainCpi_list]
    return trainDataSet#[[smiles, sequence, interaction],.....]
def getTestProteinList(testFoldPath):#获取测试集中蛋白质名称列表
    testProteinList = readLinesStrip(open(testFoldPath).readlines())[0].split()
    return testProteinList#['kpcb_2i0eA_full','fabp4_2nnqA_full',....]
def getSeqContactDict(contactPath,contactDictPath):# make a seq-contactMap dict 
    contactDict = open(contactDictPath).readlines()
    seqContactDict = {}
    for data in contactDict:
        seq,contactMapName = data.strip().split(':')
        _,contactMap = getProtein(contactPath,contactMapName)
        contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
        feature2D = np.expand_dims(contactmap_np, axis=0)
        feature2D = torch.FloatTensor(feature2D)    
        seqContactDict[seq] = feature2D
    return seqContactDict
def getLetters(path):#
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars
def getDataDict(testProteinList,activePath,decoyPath,contactPath):
    #为每个蛋白质构造对应的测试样本，包含其活性分子、虚假分子（decoys）和标签,最终得到一个字典 {蛋白质名称: 样本列表}
    dataDict = {}
    for x in testProteinList:
        xData = []
        protein = x.split('_')[0]
        proteinActPath = activePath+"/"+protein+"_actives_final.ism"
        proteinDecPath = decoyPath+"/"+protein+"_decoys_final.ism"
        act = open(proteinActPath,'r').readlines()
        dec = open(proteinDecPath,'r').readlines()
        actives = [[x.split(' ')[0],1] for x in act] ######
        decoys = [[x.split(' ')[0],0] for x in dec]# test
        seq = getProtein(contactPath,x,contactMap = False)
        for i in range(len(actives)):
            xData.append([actives[i][0],seq,actives[i][1]])
        for i in range(len(decoys)):
            xData.append([decoys[i][0],seq,decoys[i][1]])
        dataDict[x] = xData
    return dataDict

"""批处理函数"""
def my_collate(batch):#按 batch 整理 SMILES、接触图、标签、序列
    smiles = [item[0] for item in batch]
    contactMap = [item[1] for item in batch]
    label = [item[2] for item in batch]
    seq = [item[3] for item in batch]
    return [smiles, contactMap, label,seq]

"""辅助日志"""
def time_log(s):#输出带时间戳的日志信息
    print('%s-%s'%(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),s))
    return