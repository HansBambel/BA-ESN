import numpy as np


def generateParityData(N=10000, n=3, randomstate=np.random.RandomState(42)):

    rng = randomstate
    bits = rng.randint(2, size=N)

    # Calculate n-parity
    parity = n*[0]
    for i in range(n, N):
            odd = False
            for j in range(i-n,i):
                if bits[j] == 1:
                    odd = not odd
            parity.append(1 if odd else 0)
    return bits, np.array(parity)


bits, parity = generateParityData()

# print("len(bits):",len(bits), "len(parity)",len(parity))
# print(bits[0:20])
# print(parity[0:20])
