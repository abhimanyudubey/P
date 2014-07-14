#!/usr/bin/python

# Server program

#!/usr/bin/python
from socket import *
import sys

host = sys.argv[1]
port = int(sys.argv[2])
addr = (host,port)
buf = 1024
UDPSock = socket(AF_INET,SOCK_DGRAM)
UDPSock.bind(addr)

numCollections=5
maxCollectionLen=300000

lockCollections = []
for i in range(numCollections): lockCollections.append({})
currentLockCollection = 0

# Receive messages
while 1:
    data,addr = UDPSock.recvfrom(buf)

    keyFound = False
    for i in range(numCollections):
        if lockCollections[i].has_key(data):
            keyFound = True

    if keyFound:
        UDPSock.sendto('stop', addr)
    else:
        UDPSock.sendto('go', addr)
        print "lock '%s' granted to %s" % (data, addr)
        if len(lockCollections[currentLockCollection]) == maxCollectionLen:
            currentLockCollection = (currentLockCollection + 1) % numCollections
            lockCollections [currentLockCollection] = {}
            print "** flushing lock collection %d" % currentLockCollection
        lockCollections [currentLockCollection] [data] = 1

# Close socket
UDPSock.close()
