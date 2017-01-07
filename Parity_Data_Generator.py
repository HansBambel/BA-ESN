import numpy as np


def generateParityData(N=10000,
                       n=3,
                       zero=[1, 0, 1, 0, 1, 0],
                       one= [1, 1, 1, 0, 0, 0],
                       randomstate=np.random.RandomState(42)):

    if len(zero)!=len(one):
        raise ValueError("Encodings for Zero and One need to be the same length")
    rng = randomstate
    bits = rng.randint(2, size=N)

    # How to encode 0 and 1 for the target?
    target = 3 * len(zero) * [0]

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
    return bits, np.array(parity), np.array(target)


# bits, parity = generateParityData()
# print("len(bits):",len(bits), "len(parity)",len(parity))
# print(bits[0:20])
# print(parity[0:20])
