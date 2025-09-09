x = [0, 1, 2, 3, 3, 5, 5, 6, 7, 7, 10, 5]
y = [96, 85, 82, 74, 95, 68, 76, 58, 65, 75, 50, 84]

n = len(x)
sumX = 0
sumY = 0
sumX2 = 0
sumXY = 0
for i in range(n):
    sumX += x[i]
    sumY += y[i]
    sumX2 += x[i] * x[i]
    sumXY += x[i] * y[i]

xBar = sumX / n
yBar = sumY / n

print("n =", n)
print("sumX =", sumX, "sumY =", sumY)
print("sumX2 =", sumX2, "sumXY =", sumXY)
print("xBar =", xBar, "yBar =", yBar)

num = n * sumXY - sumX * sumY
den = n * sumX2 - sumX ** 2
m = num / den
c = yBar - m * xBar

print("\nSlope m = ", m)
print("\nIntercept c = ", c)

hours = int(input("\nEnter study hours to predict marks: "))
predicted_marks = m * hours + c
print(f"Predicted marks for {hours} hours of study = {predicted_marks}")

