import cv2
import numpy as np

sizeofscr = 1920

blank_image = np.zeros((1080,sizeofscr,3), np.uint8)

trackslist   =  [[122.1618423461914, 354.92059326171875, 899.4300537109375, 690.5003051757812, 0.0, 1.0],
                 [196.60336303710938, 77.13977813720703,  957.075927734375, 217.147216796875,  1.0, 2.0]]

originaldets =  [[     119.72,      353.91,      905.59,      686.94,           0],
                 [     196.19,      77.038,      957.28,      217.21,           1]]
track_id =  1

tracking_id = int(track_id) - 1

det1 =   trackslist[tracking_id]

det2 =  [     119.72,      353.91,      905.59,      686.94,           0]

cv2.rectangle(blank_image,(int(det2[0]),int(det2[1])),(int(det2[2]),int(det2[3])),(0,0,255),2)

p1 = [int((det1[0] + det1[2])/2) , int((det1[1] + det1[3])/2)]
p2 = [int((det2[0] + det2[2])/2) , int((det2[1] + det2[3])/2)]
print("p1 = ", p1)
print("p2 = ", p2)
det2w = int(det2[2] - det2[0])
det2h = int(det2[3] - det2[1])
print("Width and height = ", det2w, " , ", det2h)
'''
p1 = [100,220]
p2 = [120,200]'''

dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
print("distance = ", dis)

blank_image = cv2.circle(blank_image, p1, 1, (255, 255, 255), 1)
blank_image = cv2.circle(blank_image, p2, 1, (255, 255, 255), 1)

fx = abs(p1[0] - p2[0])
fy = abs(p1[1] - p2[1])

if (p1[0] > p2[0]):
        fx = p2[0] - fx
else:
        fx = fx + p2[0]

if (p1[1] > p2[1]):
        fy = p2[1] - fy
else:
        fy = fy + p2[1]

blank_image = cv2.line(blank_image,p1,p2,(0,0,255),2)
blank_image = cv2.line(blank_image,(int(fx),int(fy)),p2,(0,255,0),2)

dis2 = ((int(fx) - p2[0]) ** 2 + (int(fy) - p2[1]) ** 2) ** 0.5
print("distance2 = ", dis2)

print("point3's x = ", int(fx))
print("point3's y = ", int(fy))

det3 = det2
#print(type(det3))
print("original det = ", det3)
det3[0] = int(fx)-(det2w/2)
det3[1] = int(fy)-(det2h/2)
det3[2] = int(fx)+(det2w/2)
det3[3] = int(fy)+(det2h/2)
print("Point 3  box = ", det3)
newdets = originaldets
newdets[det3[4]] = det3
print("Newdets = ", newdets)

cv2.rectangle(blank_image,(int(det3[0]),int(det3[1])),(int(det3[2]),int(det3[3])),(0,255,0),2)

track3 = det1
print("original track = ", track3)
track3[0] = det3[0]
track3[1] = det3[1]
track3[2] = det3[2]
track3[3] = det3[3]
print("New  3   track = ", track3)
trackslist[tracking_id] = track3
print("trackslist = ", trackslist)

cv2.imshow("test",blank_image)
cv2.waitKey(0)