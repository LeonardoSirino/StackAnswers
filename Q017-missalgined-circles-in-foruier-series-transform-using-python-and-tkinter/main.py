import math
import time
from tkinter import *

CIRCLES_NUMBER = 4

root = Tk()
myCanvas = Canvas(root, width=1300, height=750)
myCanvas.pack()
myCanvas.configure(bg="#0A2239")

global x, y, lines, xList, yList


rList = [200]
n = 3
for i in range(0, CIRCLES_NUMBER):
    rList.append(rList[0] / n)
    n += 2

print(rList)

num = 250/sum(rList)

for i in range(0, CIRCLES_NUMBER):
    rList[i] = rList[i]*num


x = 0
y = 0
lines = []
circles = []

centerXList = [300] + [0] * CIRCLES_NUMBER
centerYList = [300] + [0] * CIRCLES_NUMBER

xList = [0] * CIRCLES_NUMBER
yList = [0] * CIRCLES_NUMBER

waveLines = []
wavePoints = []
con = 0


endCoord = []
for i in range(0, CIRCLES_NUMBER):
    endCoord.append([0, 0])

lastX = 0
lastY = 0

count = 0

randlist = []
n = 1
for i in range(0, CIRCLES_NUMBER):
    randlist.append(200/n)
    n += 2


def createCircle(x, y, r, canvasName):
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvasName.create_oval(x0, y0, x1, y1, width=r/50, outline="#094F9A")


def updateCircle(i):
    newX = endCoord[i-1][0]
    newY = endCoord[i-1][1]

    centerXList[i] = newX
    centerYList[i] = newY

    x0 = newX - rList[i]
    y0 = newY - rList[i]
    x1 = newX + rList[i]
    y1 = newY + rList[i]

    myCanvas.coords(circles[i], x0, y0, x1, y1)


def circleWithLine(i):
    global line, lines
    circle = createCircle(centerXList[i], centerYList[i], rList[i], myCanvas)
    circles.append(circle)
    line = myCanvas.create_line(
        centerXList[i], centerYList[i], centerXList[i], centerYList[i], width=2, fill="#1581B7")
    lines.append(line)


def update(i, x, y):
    endCoord[i][0] = x + rList[i]*math.cos(xList[i])
    endCoord[i][1] = y + rList[i]*math.sin(yList[i])

    myCanvas.coords(lines[i], x, y, endCoord[i][0], endCoord[i][1])
    xList[i] += math.pi / randlist[i]
    yList[i] += math.pi / randlist[i]


def lineBetweenTwoPoints(x, y, x2, y2):
    line = myCanvas.create_line(x, y, x2, y2, fill="white")
    return line


def lineForWave(y1, y2, y3, y4, con):
    l = myCanvas.create_line(700+con, y1, 702+con, y2,
                             704+con, y3, 706+con, y4, smooth=1, fill="white")
    waveLines.append(l)


for i in range(0, CIRCLES_NUMBER):
    circleWithLine(i)

myCanvas.create_line(700, 20, 700, 620, fill="black", width=3)
myCanvas.create_line(700, 300, 1250, 300, fill="red")

myCanvas.create_line(0, 300, 600, 300, fill="red", width=0.5)
myCanvas.create_line(300, 0, 300, 600, fill="red", width=0.5)

while True:
    for i in range(0, len(lines)):
        update(i, centerXList[i], centerYList[i])

    for i in range(1, len(lines)):
        updateCircle(i)

    if count >= 8:
        lineBetweenTwoPoints(lastX,
                             lastY,
                             endCoord[i][0],
                             endCoord[i][1])

        if count % 6 == 0 and con < 550:
            lineForWave(wavePoints[-7], wavePoints[-5],
                        wavePoints[-3], wavePoints[-1], con)
            con += 6

    wavePoints.append(endCoord[i][1])
    myCanvas.update()

    lastX = endCoord[i][0]
    lastY = endCoord[i][1]

    if count != 108:
        count += 1
    else:
        count = 8

    time.sleep(0.01)
