nums = [3,2,1,5,6,4]
k = 2
import heapq
nums = [-i for i in nums]
heapq.heapify(nums)
print(nums)
j = 1
while k >= j:
    if k == j:
        print(-(heapq.heappop(nums)))
    else:
        heapq.heappop(nums)
    j += 1