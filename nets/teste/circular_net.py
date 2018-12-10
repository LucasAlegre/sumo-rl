import argparse
import os
import sys
import math
import subprocess

numPoints = 128

def nextId():
    id = 1
    while True:
        yield(id)
        id += 1

nodesId = nextId()
edgesId = nextId()

class Circular:
    def __init__(self, x, y, r, flatRate=1):
        self.x = x
        self.y = y
        self.r = r
        self.flatRate = flatRate
        self.nodes = []
        self.edges = []
        angle = 2*math.pi/numPoints
        shape = [(math.cos(i * angle) * self.r + self.x, self.flatRate * math.sin(i * angle) * self.r + self.y) for i in range(numPoints+1)]
        self.nodes.append('''<node id="{}" x="{}" y="{}"/>'''.format(next(nodesId), shape[0][0], shape[0][1]))
        for i in range(4,numPoints+1,4):
            currentId = next(nodesId)
            self.nodes.append('''<node id="{}" x="{}" y="{}"/>'''.format(currentId, shape[i][0], shape[i][1]))
            self.edges.append('''<edge id="{}" from="{}" to="{}" shape="{}" numLanes="2"/>'''.format(next(edgesId), currentId-1, currentId, " ".join(["%.2f,%.2f" % x for x in shape[i-4:i+1]])))
        
	
    def getNodes(self):
        return "\n".join(self.nodes)        

    def getEdges(self):
        return "\n".join(self.edges)


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="")

    prs.add_argument("-o", dest="file", required=True)
    args = prs.parse_args()
    filename = args.file
    edgefile = filename + '.edg.xml'
    nodfile = filename + '.nod.xml'
    netfile = filename + '.net.xml'

    c1 = Circular(-50, 20, 100, 1.5)
    c2 = Circular(50, 20, 100, 1.5)
    c3 = Circular(0, 200, 150, 1)
    c4 = Circular(0, 350, 175, 0.5)

    with open(nodfile, "w+") as output:
        print("<nodes>", file=output)
        print(c1.getNodes(), file=output)
        print(c2.getNodes(), file=output)
        print(c3.getNodes(), file=output)
        print(c4.getNodes(), file=output)
        print("</nodes>", file=output)

    with open(edgefile, "w+") as output:
        print("<edges>", file=output)
        print(c1.getEdges(), file=output)
        print(c2.getEdges(), file=output)
        print(c3.getEdges(), file=output)
        print(c4.getEdges(), file=output)
        print("</edges>", file=output)

    subprocess.call("netconvert -n {} -e {} -o {}".format(nodfile, edgefile, netfile), shell=True)


