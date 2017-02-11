import numpy as np


def generateParityData(N=10000,
                       n=3,
                       timescale=16,
                       randomstate=np.random.RandomState(42)):

    if timescale < 1:
        raise ValueError("Timescale can't be lower than 1")

    rng = randomstate
    bits = rng.randint(2, size=N)
    # first (unimportant) targets
    target = n*[0]
    # Calculate n-parity
    parity = n*[0]
    for i in range(n, N):
            odd = False
            for j in range(i-n,i):
                if bits[j] == 1:
                    odd = not odd
            if odd:
                parity.append(1)
                target.append(1)
            else:
                parity.append(0)
                target.append(0)
            # parity.append(1 if odd else 0)
    bits = np.array(bits)
    parity = np.array(parity)
    target = np.array(target)

    ext_bits = np.repeat(bits, timescale).reshape(-1, 1)
    ext_parity = np.repeat(parity, timescale).reshape(-1, 1)
    ext_target = np.repeat(target, timescale).reshape(-1, 1)
    return ext_bits, ext_parity, ext_target

# import time
# start_time = time.time()
# bits, parity, target = generateParityData()
# print("--- %s seconds ---" % (time.time() - start_time))
# print(np.shape(bits),np.shape(parity),np.shape(target))
# print("len(bits):",len(bits), "len(parity)",len(parity))
# print(bits[66600:66680].reshape(-1))
# print(parity[66600:66680].reshape(-1))
