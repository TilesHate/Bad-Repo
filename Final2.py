proccess = input("Are you sure you want to procced? <y|n> : ")

if proccess.lower() != "y": exit()

import os
import sys
import cv2
import numpy as np
import struct
from time import perf_counter
from time import sleep
from math import ceil
from math import modf

cap = cv2.VideoCapture("bad_apple.mp4")

if not cap.isOpened(): print("Ran to an error trying to capture the video!"); exit()

def readFrame(frame) -> list:
    encodedRows = bytearray()

    for row in frame:
        diffs = np.diff(row)
        changeIndices = np.where(diffs != 0)[0] + 1

        runLengths = np.diff(np.concatenate(([0], changeIndices, [len(row)])))
        runValues = row[np.concatenate(([0], changeIndices))]

        for val, length in zip(runValues, runLengths):
            encodedRows.append(int(val))
            encodedRows.extend(struct.pack(">H", int(length)))

    return encodedRows

def decodeFrame(encoded: bytearray, width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width), dtype=np.uint8)

    i = 0
    row = 0
    col = 0

    while i < len(encoded) and row < height:
        # Read value
        val = encoded[i]
        i += 1

        # Read run length (2 bytes, big-endian)
        length = struct.unpack(">H", encoded[i:i+2])[0]
        i += 2

        # Fill pixels
        frame[row, col:col+length] = val
        col += length

        # If row is filled, move to next
        if col >= width:
            row += 1
            col = 0

    return frame

def writeIntoChunks(filename, parts=11):
    with open(filename, "rb") as f:
        data = f.read()
    
    chunkSize = (len(data) + parts - 1) // parts
    base, ext = os.path.splitext(filename)

    for i in range(parts):
        start = i * chunkSize
        end = start + chunkSize
        chunk = data[start:end]

        if not chunk: break

        chunkName = f"{base}_part{i+1}{ext}"

        with open(chunkName, "wb") as chunkBinary:
            chunkBinary.write(chunk)
        
        print(f"Saved '{chunkName}' chunk <length '{len(chunk)}bytes'>")
    
    print(f"Successfully splitted '{filename}' into '{parts}' chunks!")

try:
    startTime = perf_counter()
    endTime = None

    with open("bad_frames.bin", "wb") as f:

        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            read, frame = cap.read()

            if not read: break

            binary = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 128).astype(np.uint8)

            adjustFrame = cv2.resize(
                binary,
                (frame.shape[1] // 4, frame.shape[0] // 4),
                interpolation=cv2.INTER_AREA
            )

            result = readFrame(adjustFrame)

            f.write(result)

    writeIntoChunks("bad_frames.bin", 11)

    endTime = perf_counter()

except Exception as e: print(f"Closed due to an error!! : {e}")
else:
    secs, mins = modf(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / ceil(cap.get(cv2.CAP_PROP_FPS)) / 60)
    print(
        f"Successfully processed Bad Apple!! with the length of {int(mins)}:{round(secs * 60)} "
        f"<with '{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames'> "
        f"<running at '{round(cap.get(cv2.CAP_PROP_FPS))} FPS'> "
        f"<took '{(endTime - startTime):.5f}s'>"
    )
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished excuting!")
    exit()