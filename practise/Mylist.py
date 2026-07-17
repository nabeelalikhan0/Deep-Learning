class MyList:

    def __init__(self, *args):
        self.list = list(args)

    def __repr__(self):
        return str(self.list)

    def view(self, rows, cols):

        if rows == -1:
            rows = len(self.list) // cols

        elif cols == -1:
            cols = len(self.list) // rows

        if rows * cols != len(self.list):
            raise ValueError("Invalid shape")

        result = []

        for i in range(rows):
            start = i * cols
            end = start + cols
            result.append(self.list[start:end])

        return result
    


m1 = MyList(1,2,3,4,5,6,1,2,3,4,5,6)
print(m1.view(-1,6))