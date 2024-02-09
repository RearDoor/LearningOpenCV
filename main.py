# Python
import numpy as np;
import cv2; 

# Video Inference
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

vid_cap = cv2.VideoCapture('video2.mp4')

frame_width = int(vid_cap.get(3))
frame_height = int(vid_cap.get(4))
fps = vid_cap.get(cv2.CAP_PROP_FPS)
vid_write = cv2.VideoWriter('output2.mp4', fourcc, fps, (frame_width, frame_height))


all_rows = open('synset.txt').read().strip().split("\n")
classes = [r[r.find(' ') + 1:] for r in all_rows] # WHYY?

net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.proto.txt','bvlc_googlenet.caffemodel')

if vid_cap.isOpened() == False:
    print('Failed to open video')

while True:
    ret, frame = vid_cap.read()

    if ret:
        cv2.imwrite('output.jpeg',frame)
        blob = cv2.dnn.blobFromImage(frame,1,(224,224))

        net.setInput(blob)

        out = net.forward()

        index = "{} {}"

        row = 1
        for i in np.argsort(out[0])[::-1][:5]:
            txt =  index.format(classes[i], out[0][i]*100)
            cv2.putText(frame, txt, (0,25 + 40*row), cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),2)
            row = row + 1

        
        vid_write.write(frame)

    else:
        break

vid_cap.release()
cv2.destroyAllWindows()

















## FIRST INFERENCE


# all_rows = open('synset.txt').read().strip().split("\n")
# classes = [r[r.find(' ') + 1:] for r in all_rows] # WHYY?

# net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.proto.txt','bvlc_googlenet.caffemodel')


# ## Images
# img = cv2.imread('output.jpeg')

# blob = cv2.dnn.blobFromImage(img,1,(224,224))
# net.setInput(blob)
# out = net.forward()

# index = np.argsort(out[0])[::-1][:5]
# print (index)
# for (i,v) in enumerate(index):
#     print(i+1,classes[v],v,out[0][v])

# print(img.shape)
# print(type(img))

# b = img[:,:,0]
# g = img[:,:,1]
# r = img[:,:,2]


# cv2.imwrite('output_b.jpeg',b)
# cv2.imwrite('output_g.jpeg',g)
# cv2.imwrite('output_r.jpeg',r)









# # Video
# # Define the codec using VideoWriter_fourcc() and create a VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

# vid_cap = cv2.VideoCapture('video2.mp4')




# # Get the video's width, height, and frames per second (fps)
# frame_width = int(vid_cap.get(3))
# frame_height = int(vid_cap.get(4))
# fps = vid_cap.get(cv2.CAP_PROP_FPS)

# vid_write = cv2.VideoWriter('output2.mp4', fourcc, fps, (frame_width, frame_height))



# while True:


#     ret, frame = vid_cap.read()

#     if ret:
#         cv2.imwrite('output.jpeg',frame)
#         break
#         b = np.copy(frame)
#         g = np.copy(frame)
#         r = np.copy(frame)

#         b[:,:,0] = 0
#         vid_write.write(b)
#         b[:,:,1] = 0
#         vid_write.write(b)
#         g[:,:,1] = 0
#         vid_write.write(g)
#         g[:,:,2] = 0
#         vid_write.write(g)
#         r[:,:,2] = 0
#         vid_write.write(r)
#         r[:,:,0] = 0
#         vid_write.write(r)

#     else:
#         break
    
# vid_cap.release()
# vid_write.release()
# cv2.destroyAllWindows()







