
"""
Input Recorder - Record and play back keyboard and mouse input.
Copyright (C) 2022  Zuzu_Typ

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import winput, re

def vk_code_to_key_name(vk_code):
    name = winput.vk_code_dict.get(vk_code, "VK_UNKNOWN")

    if name.startswith("VK_"):
        name = name[3:]
    
    special_left_key_match = re.match("L(CONTROL|MENU|WIN|SHIFT)", name)

    if special_left_key_match:
        name = "LEFT " + special_left_key_match.group(1)

    special_right_key_match = re.match("R(CONTROL|MENU|WIN|SHIFT)", name)

    if special_right_key_match:
        name = "RIGHT " + special_right_key_match.group(1)

    name = name.replace("_", " ")

    name = " ".join((word.capitalize() for word in name.split(" ")))

    oem_match = re.fullmatch("(.*)OEM(.*)", name, re.IGNORECASE)

    if oem_match:
        name = oem_match.group(1) + "OEM" + oem_match.group(2)
    
    return name

def mouse_button_to_name(button):
    return "Left" if button == winput.LMB else \
           "Right" if button == winput.RMB else \
           "Middle" if button == winput.MMB else \
           "X1" if button == winput.XMB1 else \
           "X2"

# 按像素比较相似度
def _check_color(full_img, regn_img):
    import cv2
    DIFF = 20
    regn_data = cv2.cvtColor(regn_img, cv2.COLOR_BGR2RGB)
    full_data = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
    CX = regn_data.shape[0]
    CY = regn_data.shape[1]
    if CX != full_data.shape[0] or CY != full_data.shape[1]:
        return False
    diff_num = 0
    for x in range(0, CX):
        for y in range(0, CY):
            regn_pixel = regn_data[y, x]
            full_pixel = full_data[y, x]
            diff_r = int(full_pixel[0]) - int(regn_pixel[0])
            diff_g = int(full_pixel[1]) - int(regn_pixel[1])
            diff_b = int(full_pixel[2]) - int(regn_pixel[2])
            if abs(diff_r) > DIFF or abs(diff_g) > DIFF or abs(diff_b) > DIFF:
                diff_num=diff_num+1
    print(diff_num)            
    return diff_num < (CX * CY) * 0.1

# 比较两个图片文件是否相似
def is_similar_image(file1, file2):
    import cv2

    sift = cv2.xfeatures2d.SIFT_create()
    full_img = cv2.imread(file1, 0)
    regn_img = cv2.imread(file2, 0)

    kp_full, desc_full = sift.detectAndCompute(full_img, None)
    kp_regn, desc_regn = sift.detectAndCompute(regn_img, None)

    if len(kp_full) == 0 or len(kp_regn) == 0:
        if _check_color(full_img, regn_img):
            return True
        return False

    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(desc_full, desc_regn)
    matches = sorted(matches, key=lambda x: x.distance)
    match_position = kp_full[matches[0].queryIdx].pt if len(matches) > 0 else (-1,-1)
    # match_distance = matches[0].distance
    # match_dist2 = pow(match_position[0] - mouse_x0,2) + pow(match_position[1] - mouse_y0,2)
    # for mm in matches:
    #     if abs(mm.distance - match_distance) > 3:
    #         continue
    #     pt = kp_full[mm.queryIdx].pt
    #     dist2 = pow(pt[0] - mouse_x0,2) + pow(pt[1] - mouse_y0,2)
    #     if dist2 < match_dist2:
    #         match_dist2 = dist2
    #         match_position = pt

    print("match distance=", matches[0].distance, "pt=", match_position)

    if matches[0].distance < 150:
        return True
    return False

def is_similar_image2(full_img, regn_img):
    import cv2

    sift = cv2.xfeatures2d.SIFT_create()

    kp_full, desc_full = sift.detectAndCompute(full_img, None)
    kp_regn, desc_regn = sift.detectAndCompute(regn_img, None)

    if len(kp_full) == 0 or len(kp_regn) == 0:
        if _check_color(full_img, regn_img):
            return 0
        return 10000

    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(desc_full, desc_regn)
    matches = sorted(matches, key=lambda x: x.distance)
    match_position = kp_full[matches[0].queryIdx].pt if len(matches) > 0 else (-1,-1)
    # match_distance = matches[0].distance
    # match_dist2 = pow(match_position[0] - mouse_x0,2) + pow(match_position[1] - mouse_y0,2)
    # for mm in matches:
    #     if abs(mm.distance - match_distance) > 3:
    #         continue
    #     pt = kp_full[mm.queryIdx].pt
    #     dist2 = pow(pt[0] - mouse_x0,2) + pow(pt[1] - mouse_y0,2)
    #     if dist2 < match_dist2:
    #         match_dist2 = dist2
    #         match_position = pt

    print("match distance=", matches[0].distance, "pt=", match_position)

    return matches[0].distance
