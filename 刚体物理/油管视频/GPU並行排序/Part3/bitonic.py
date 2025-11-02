import numpy as np
def compare_and_swap(arr, indices, stride):
   
        new_indices = indices.copy()
        for i in range(0, len(indices), stride * 2):
            for j in range(stride):
                a = new_indices[i + j]
                b = new_indices[i + j + stride]
                if arr[a] > arr[b]:
                    new_indices[i + j], new_indices[i + j + stride] = b, a
        return new_indices

def bitonic_sort_steps(arr):
        """
        输入长度为 2^k 的数组，输出排序过程中每一轮索引变换
        """
        n = len(arr)
        assert (n & (n - 1)) == 0, "数组长度必须为2的幂"

        indices = np.arange(n)
        steps = []

        k = int(np.log2(n))
        for stage in range(k):
          
            stride = 2**(k-1-stage)
            indices = compare_and_swap(arr, indices, stride)
            steps.append(indices.copy())

        return np.array(steps)
    