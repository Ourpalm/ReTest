def check_color(full_img, regn_img):
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

def test_match(file1,file2):
    import cv2
    sift = cv2.xfeatures2d.SIFT_create()
    full_img = cv2.imread(file1, 0)
    regn_img = cv2.imread(file2, 0)

    kp_full, desc_full = sift.detectAndCompute(full_img, None)
    kp_regn, desc_regn = sift.detectAndCompute(regn_img, None)

    if len(kp_full) == 0 or len(kp_regn) == 0:
        if check_color(full_img, regn_img):
            print("1")
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


match = test_match('match_full.png','match_regn.png')
print(match)