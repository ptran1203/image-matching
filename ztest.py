def minimumBribes(q):
    for i in range(len(q)):



inputs = '1 2 5 3 4 7 8 6'
inputs = map(int, inputs.split())
minimumBribes(inputs)


"""
1 2 3 4 5
1 2 3 5 4 --> 1
1 2 5 3 4 --> 1
2 1 5 3 4 --> 1

-> 3
"""