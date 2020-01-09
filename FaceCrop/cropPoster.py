#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[2]:


img=cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)


# In[3]:


img.shape


# In[4]:


cv2.imshow("img",img)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# In[5]:


faces = face_cascade.detectMultiScale(img, 1.3, 5)


# In[6]:


def getRes(aspect,original):
    if aspect<original[0]/original[1]:
        return(int(original[1]*aspect),original[1])
    else:
        return(original[0],int(original[0]/aspect))


# In[7]:


if(sys.argv[2]==""):
    aspectRatio=1
else:
    aspectRatio=float(sys.argv[2])


# In[15]:



for (x,y,w,h) in faces:
    width,height=getRes(1,img.shape)
    start_x=int(x+w/2-width/2)
    start_y=int(y+h/2-height/2)
    if(start_y<0):
        start_y=0
    if(start_x<0):
        start_x=0
    print(start_x,start_y,width,height)
    #cv2.rectangle(img,(int(start_x),int(start_y)),(int(start_x+width),int(start_y+height)),(255,255,0),2)
    #cv2.imshow('img',img)
    #cv2.waitKey(3000)
    cropped=img[start_y:start_y+height,start_x:start_x+width]
    cv2.imshow('cropped',cropped)
    cv2.waitKey(3000)
    cv2.imwrite(sys.argv[1].split(".")[0]+"_cropped.jpg",cropped)
cv2.destroyAllWindows()




