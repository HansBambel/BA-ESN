import numpy as np


def generateParityData(N=10000,
                       n=3,
                       # zero=[1, 0, 1, 0, 1, 0],
                       # one= [1, 1, 1, 0, 0, 0],
                       zero=None,
                       one= None,
                       randomstate=np.random.RandomState(42)):

    if  zero is None:
        # zero = [0, -0.5, 0, -0.5, 0, -0.5, 0, -0.5, 0, -0, 0, -0.5]
        zero = [0, 0.25, 0, -0.25,
                0, 0.25, 0, -0.25,
                0, 0.25, 0, -0.25,
                # 0, 0.25, 0, -0.25,
                0, 0.25, 0, -0.25]
    if one is None:
        # one = [0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0]
        # konstanter output
        one = [0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0,
               # 0, 0, 0, 0,
               0, 0, 0, 0]

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
    bits = np.array(bits)
    parity = np.array(parity)
    target = np.array(target).reshape(-1,1)

    ext_bits = np.repeat(bits, len(zero)).reshape(-1, 1)
    ext_parity = np.repeat(parity, len(zero)).reshape(-1, 1)
    return ext_bits, ext_parity, target

# import time
# start_time = time.time()
# bits, parity, target = generateParityData()
# print("--- %s seconds ---" % (time.time() - start_time))
# print(np.shape(bits),np.shape(parity),np.shape(target))
# print("len(bits):",len(bits), "len(parity)",len(parity))
# print(bits[66600:66680].reshape(-1))
# print(parity[66600:66680].reshape(-1))
