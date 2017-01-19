import numpy as np


def generateParityData(N=10000,
                       n=3,
                       # zero=[1, 0, 1, 0, 1, 0],
                       # one= [1, 1, 1, 0, 0, 0],
                       zero=None,
                       one= None,
                       randomstate=np.random.RandomState(42)):

    if  zero is None:
        zero = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
                # 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
    if one is None:
        one = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
                # 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

    if len(zero)!=len(one):
        raise ValueError("Encodings for Zero and One need to be the same length")
    rng = randomstate
    bits = rng.randint(2, size=N)

    # first (unimportant) targets
    target = n * len(zero) * [0]

    # Calculate n-parity
    parity = n*[0]
    for i in range(n, N):
            odd = False
            for j in range(i-n,i):
                if bits[j] == 1:
                    odd = not odd
            if odd:
                parity.append(1)
                target = target + one
            else:
                parity.append(0)
                target = target + zero
            # parity.append(1 if odd else 0)
    bits = np.array(bits).reshape(-1,1)
    parity = np.array(parity).reshape(-1,1)
    target = np.array(target).reshape(-1,1)

    ext_bits, ext_parity = [], []
    for i in range(len(bits)):
        for j in range(len(zero)):
            ext_bits.append(bits[i])
            ext_parity.append(parity[i])
    ext_bits = np.array(ext_bits)
    ext_parity = np.array(ext_parity)
    return ext_bits, ext_parity, target


# bits, parity, target = generateParityData()
# print(np.shape(bits),np.shape(parity),np.shape(target))
# print("len(bits):",len(bits), "len(parity)",len(parity))
# print(bits[0:20])
# print(parity[0:20])
