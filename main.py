import json
import math
print()


def absolute(num):
    if(num >= 0):
        return num
    else:
        return num * (-1)


def cost(theta_0, theta_1, inputs, outputs, learningRate):
    costSum_0 = 0
    costSum_1 = 0
    n = len(inputs)

    for x, y in zip(inputs, outputs):
        guess = theta_0 + theta_1 * x
        error = y - guess

        costSum_0 += (learningRate*1000) * (error)
        costSum_1 += (learningRate / n) * (error * x)

    return costSum_0, costSum_1


def gradientDescent(inputs, outputs):
    # H(x) = theta_0 + theta_1 * x

    theta_0 = 0
    theta_1 = 1

    n = len(inputs)
    learningRate = 0.000005

    while(True):
        temp_0, temp_1 = cost(theta_0, theta_1, inputs, outputs, learningRate)

        if(absolute(temp_0) < 0.0001 and absolute(temp_1) < 0.0001):
            break
        else:
            theta_0 += temp_0
            theta_1 += temp_1

        print(f"theta_0: {theta_0},  theta_1: {theta_1}")

    return theta_0, theta_1


def main():
    with open("data.json") as f:
        data = json.load(f)["housesPrice"]

    inputs = []
    outputs = []

    for d in data:
        inputs.append(d["x"])
        outputs.append(d["y"])

    theta_0, theta_1 = gradientDescent(inputs, outputs)
    print(f"H(x) = {theta_0} + {theta_1} * X")


if __name__ == '__main__':
    main()
    print()
