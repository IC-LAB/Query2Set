# coding: utf-8
import math
import random
import numpy as np
import cv2

def is_legal(legal_region, center, rs, threshold=0.9):
    cir_region = np.zeros_like(legal_region, dtype=np.uint8)
    cv2.circle(cir_region, (center[0],center[1]), rs, 255, -1)
    _, cir_region = cv2.threshold(cir_region, 127, 255, cv2.THRESH_BINARY)
    and_region = cv2.bitwise_and(legal_region, legal_region, mask=cir_region)
    overlap_ratio = np.nonzero(and_region)[0].shape[0] / (math.pi*rs*rs)
    return (overlap_ratio>=threshold)

def rotate_xy(x,y,angle,cx,cy):
    x_new = (x-cx)*math.cos(angle) - (y-cy)*math.sin(angle) + cx
    y_new = (x-cx)*math.sin(angle) + (y-cy)*math.cos(angle) + cy
    return [int(x_new), int(y_new)]

def plot_block(image, block, color=(0,0,255)):
    center_x = block['center'][0]
    center_y = block['center'][1]
    hs = int(block['x'].shape[0]/2)
    rs = int(hs*math.sqrt(2))
    block_pts = [[block['center'][0]-hs,block['center'][1]-hs], 
                 [block['center'][0]-hs,block['center'][1]+hs], 
                 [block['center'][0]+hs,block['center'][1]+hs], 
                 [block['center'][0]+hs,block['center'][1]-hs]]
    block_pts = [rotate_xy(pt[0],pt[1],block['angle'],block['center'][0],block['center'][1]) for pt in block_pts]

    cv2.line(image, (center_x-25,center_y), (center_x+25,center_y), color, 2)
    cv2.line(image, (center_x,center_y-25), (center_x,center_y+25), color, 2)
    cv2.circle(image, (center_x,center_y), rs, color, 2)
    cv2.polylines(image, np.array([block_pts]), isClosed=True, color=color, thickness=2)

def get_si(image, legal_region, center_candidate, q, mt=10, debug=False):
    bs = q['x'].shape[0]
    hs = int(bs/2)
    rs = int(hs*math.sqrt(2))
    si = {'x':np.zeros((bs,bs), dtype=np.uint8), 'center':[-1,-1], 'angle':-1, 'ratio':-1}
    q_pts = [[q['center'][0]-hs,q['center'][1]-hs], 
             [q['center'][0]-hs,q['center'][1]+hs], 
             [q['center'][0]+hs,q['center'][1]+hs], 
             [q['center'][0]+hs,q['center'][1]-hs]]
    q_pts = [rotate_xy(pt[0],pt[1],q['angle'],q['center'][0],q['center'][1]) for pt in q_pts]

    try_count = 1
    while try_count<mt:
        idx = random.randint(0, center_candidate[0].shape[0]-1)
        si['center'] = [center_candidate[1][idx], center_candidate[0][idx]]
        if not is_legal(legal_region, q['center'], rs, 0.8):
            try_count += 1
            continue
        si['angle'] = math.pi * random.random()
        si_pts = [[si['center'][0]-hs,si['center'][1]-hs], 
                  [si['center'][0]-hs,si['center'][1]+hs], 
                  [si['center'][0]+hs,si['center'][1]+hs], 
                  [si['center'][0]+hs,si['center'][1]-hs]]
        si_pts = [rotate_xy(pt[0],pt[1],si['angle'],si['center'][0],si['center'][1]) for pt in si_pts]

        q_mask = np.zeros_like(legal_region, dtype=np.uint8)
        cv2.fillPoly(q_mask, np.array([q_pts]), 255)
        si_mask = np.zeros_like(legal_region, dtype=np.uint8)
        cv2.fillPoly(si_mask, np.array([si_pts]), 255)
        overlap_mask = cv2.bitwise_and(q_mask, q_mask, mask=si_mask)
        overlap_ratio = np.nonzero(overlap_mask)[0].shape[0] / (bs*bs)
        if overlap_ratio<q['ratio'][0] or overlap_ratio>q['ratio'][1]:
            try_count += 1
            continue
        else:
            si['ratio'] = overlap_ratio
            rotated_si = image[si['center'][1]-bs:si['center'][1]+bs, si['center'][0]-bs:si['center'][0]+bs]
            rotate_M_si = cv2.getRotationMatrix2D((bs,bs), si['angle']*180/math.pi, 1)
            rotated_si = cv2.warpAffine(rotated_si, rotate_M_si, (bs*2, bs*2))
            si['x'] = rotated_si[hs:hs*3,hs:hs*3]
            break
    return si

def get_query2set(image, s_len=4, bs=200, ratio=[0.05,1.0], mt=10, debug=False):
    """
    生成指定重叠面积比例的query-set partial fingerprint的正样本对.

    Parameters:
    image: original fingerprint (W, H)
    s_len: number of set partial fingerprint
    bs: block size (default=200)
    ratio: valid overlap ratio (default=[0.05,1.0])
    mt: max try time
    debug: Whether to display the results of the intermediate process

    Returns:
    q: {x:(block_size, block_size, 1), center:(center_x,center_y), angle:0~pi, ratio:[ratio_min,ratio_max]}
    s: [{x:(block_size, block_size, 1), center:(center_x,center_y), angle:0~pi, ratio:0~1}*s_len]
    """

    hs = int(bs/2)
    rs = int(hs*math.sqrt(2))
    q = {'x':np.zeros((bs,bs), dtype=np.uint8), 'center':[-1,-1], 'angle':-1, 'ratio':ratio}
    s = []

    if(len(image.shape)==3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.copyMakeBorder(image,bs,bs,bs,bs,cv2.BORDER_CONSTANT,value=255)
    image_binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    image_binary_opened = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, open_kernel)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    image_binary_closed = cv2.morphologyEx(image_binary_opened, cv2.MORPH_CLOSE, close_kernel)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rs,rs))
    image_binary_eroded = cv2.erode(image_binary_closed, erode_kernel, 1)

    center_q_candidate = np.nonzero(image_binary_eroded)
    if(center_q_candidate[0].shape[0]==0):
        return q, s
    
    try_count = 1
    while try_count<mt:
        idx = random.randint(0, center_q_candidate[0].shape[0]-1)
        q['center'] = [center_q_candidate[1][idx], center_q_candidate[0][idx]]
        if not is_legal(image_binary_closed, q['center'], rs):
            try_count += 1
            continue
        q['angle'] = math.pi * random.random()
        rotated_q = image[q['center'][1]-bs:q['center'][1]+bs, q['center'][0]-bs:q['center'][0]+bs]
        rotate_M_q = cv2.getRotationMatrix2D((bs,bs), q['angle']*180/math.pi, 1)
        rotated_q = cv2.warpAffine(rotated_q, rotate_M_q, (bs*2, bs*2))
        q['x'] = rotated_q[hs:hs*3,hs:hs*3]

        cir_region = np.zeros_like(image_binary_closed, dtype=np.uint8)
        if ratio[1]<0.4:
            cv2.circle(cir_region, (q['center'][0],q['center'][1]), rs*2, 255, -1)
        elif ratio[0]>0.8:
            cv2.circle(cir_region, (q['center'][0],q['center'][1]), int(rs/4), 255, -1)
        elif ratio[0]>0.7:
            cv2.circle(cir_region, (q['center'][0],q['center'][1]), int(rs/2), 255, -1)
        elif ratio[0]>0.4:
            cv2.circle(cir_region, (q['center'][0],q['center'][1]), rs, 255, -1)
        else:
            cv2.circle(cir_region, (q['center'][0],q['center'][1]), int(rs*math.sqrt(2)), 255, -1)
        _, cir_region = cv2.threshold(cir_region, 127, 255, cv2.THRESH_BINARY)
        s_candidate_region = cv2.bitwise_and(image_binary_eroded, image_binary_eroded, mask=cir_region)
        center_s_candidate = np.nonzero(s_candidate_region)
        if(center_s_candidate[0].shape[0]<s_len):
            try_count += 1
            continue

        for _ in range(s_len):
            si = get_si(image, image_binary_closed, center_s_candidate, q, debug=debug)
            if si['ratio']<0:
                break
            else:
                s.append(si)
        
        if len(s)<s_len:
            s = []
            try_count += 1
            continue
        else:
            break
    
    if(debug):
        image_show = np.expand_dims(image, -1)
        image_show = np.tile(image_show, [1,1,3])
        plot_block(image_show, q)
        for si in s:
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            plot_block(image_show, si, color)
        cv2.imshow('Image', image_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return q, s