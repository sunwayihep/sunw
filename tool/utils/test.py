def add (a):
    a.sort()
    a += a
if __name__ == '__main__':
    a = [1, 5, 4, 3, 2]
    # print(add(a))
    add(a)
    print(a)