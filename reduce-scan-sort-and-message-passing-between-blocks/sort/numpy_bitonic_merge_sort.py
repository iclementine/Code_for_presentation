def compare_and_swap_sweep(arr, size, count, c, descending):
    # print(f"{size=}, {count=}, {c=}")
    if c <= 1:
        return

    half = dist = c // 2
    for i in range(0, size):
        # the the left side do the job
        if (i % c) >= half:
            continue

        # be stable
        if arr[i] == arr[i + dist]:
            continue

        if not descending:
            direction = i % (2 * count) < count # 升序为1， 降序为 0
        else:
            direction = i % (2 * count) >= count
        if direction == (arr[i] > arr[i + dist]):
            arr[i], arr[i + dist] = arr[i + dist], arr[i]
    
def btn_sort(arr, descending=False):
    out = np.copy(arr)
    size = np.size(arr)
    count = 2

    while count <= size:
        c = count
        while c >= 2:
            compare_and_swap_sweep(out, size, count, c, descending)
            c //= 2
        count *= 2

    return out
    

import numpy as np

x = np.random.randint(0, 1000, 32)
print(x)
y = btn_sort(x, False)
print(y)
