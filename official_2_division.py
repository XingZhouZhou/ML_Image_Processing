import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


main_path = 'F:/Travail/Harvard_RA/test_ml/'
Q1_path = os.path.join(main_path, 'To_Submit','Q1')
Q2_path = os.path.join(main_path, 'To_Submit','Q2')


os.chdir(Q1_path)
os.getcwd()

files = ["img"+str(i)+".png" for i in range(5)]
for file in files:
    
    # (0) environment
    
    new_path = os.path.join(Q2_path, 'Divided', file[:-4])
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # (1) Angle
    
    #Black_back_white_font
    th, threshed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    (cx,cy), (w,h), ang = ret
    print(' ******************************************** \n')
    print('↓ The angle of the rotated text for '+ file +' ↓ \n')
    print(ang)
    print('\n ******************************************** ')

    #White_back_black_font
    if file == 'img4.png':
        thresh = 170
    else:
        thresh = 150
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]   
    #plt.figure(figsize=(30, 30))
    #plt.imshow(im_bw ,cmap='gray')
    
    # (2) Rotate
    if w>h:
        w,h = h,w
        ang += 90

    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(im_bw, M, (img.shape[1], img.shape[0]),borderValue=(255,255,255),borderMode=cv2.BORDER_CONSTANT)
    #if file == 'img4.png':
        #rotated = rotated[:,:850]
        
    # (3) Plot the Dilation
    hist = cv2.reduce(rotated,1,cv2.REDUCE_AVG).reshape(-1)
    x = np.array(range(len(hist)))
    fig, ax = plt.subplots(figsize=(15,10))
    ax.plot(x, hist,color='#9F393D')
    
    ax.set(xlabel = 'Rows', ylabel='Pixel Dilation',
           title = 'Row (y-axis) Pixel Dilation for ' + file)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylim(255, 130)
    ax.grid()
    fig.savefig(os.path.join(Q2_path, "Dilation_" + file[:-4] + ".png"))
    
    print(' ******************************************** \n')
    print(' Dilation Plot for '+ file +' is saved !')
    print('\n ******************************************** ')
    
    
    plt.show(False)
    
    # (4) Division
    rotated_copy = rotated[:]
    th = 253
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
#    print('for ' +file+' :uppers[0] = ' + str(uppers[0])+ ' uppers[-1]' + str(uppers[-1])+'\n')
#    print(' length of uppers: '+ str(len(uppers))+'\n')
#    print('for ' +file+' :lowers[0] = ' + str(lowers[0])+ ' lowers[-1]' + str(lowers[-1])+'\n')
#    print(' length of lowers: '+ str(len(lowers))+'\n')
    
    for y in uppers:
        cv2.line(rotated_copy, (0,y), (W, y), (0,0,0), thickness=3)
    
    for y in lowers:
        cv2.line(rotated_copy, (0,y), (W, y), (0,0,0), thickness=1)
    
    cv2.imwrite(os.path.join(Q2_path,file[:-4]+'_lined_.png'),rotated_copy)
#    
#    n_lines = min(len(uppers), len(lowers))
#    if len(uppers) != len(lowers):
#        lines = [0]+[int((uppers[i]+lowers[i+1])/2) for i in range(n_lines)]+ [img.shape[0]]
#    else:
#        lines = [0]+[int((uppers[i]+lowers[i])/2) for i in range(n_lines)]+ [img.shape[0]]

    for i in range(len(uppers)):   
        if i+1 == len (uppers):
            break
        new_img = rotated[lowers[i]:uppers[i-len(lowers) + len(uppers)+1],:]
        cv2.imwrite(os.path.join(new_path,file[:-4] + "_div_" + str(i) + '.png'), new_img)
    
    print(' ******************************************** \n')
    print(' Row Division for '+ file +' is done !')
    print('\n ******************************************** ')
    
    
    
#plt.figure(figsize=(40, 40))
#plt.imshow(rotated[:,:850] ,cmap='gray')


new_img=cv2.imread('F:/Travail/Harvard_RA/test_ml/To_Submit/Q2/img0/img0_div_6.png')