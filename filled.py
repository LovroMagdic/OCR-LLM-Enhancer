#iterating through threshold value of 100 - 235 and calculating minimal remainder like that we get most optimal threshold which gives us best contour
for each in arr:
    each = each.replace("dataset","dataset_deskewed")
    for i in range(100, 235, 5): # i == best_thresh
        image = cv2.imread(each)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, i, 255, cv2.THRESH_BINARY)
        # cv2.imwrite('demo/test-bw.jpg', thresh)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_copy = image.copy()

        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imwrite("image-draw.jpg", image_copy)

        c = max(contours, key = cv2.contourArea)
        # print(cv2.contourArea(c))
        x,y,w,h = cv2.boundingRect(c)
        # print(w*h)

        ar.append([cv2.contourArea(c), w*h, i])
        array.append(int(w*h)-int(cv2.contourArea(c)))

    index = array.index(min(array))
    # print(ar[index])

    i = int(ar[index][2])
    # each = each.replace("dataset_deskewed","dataset")
    image = cv2.imread(each)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, i, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('demo/dataset-result/Z05353401-bw.jpg', thresh)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = image.copy()
    c = max(contours, key = cv2.contourArea)

    black_canvas = np.zeros_like(img_gray)
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.drawContours(black_canvas, c, -1, 255, cv2.FILLED) # this gives a binary mask
    each = each.replace("dataset_deskewed", "dataset_filled")
    cv2.imwrite(each, black_canvas)