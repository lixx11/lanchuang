#!/usr/bin/env python
#!coding=utf-8

from selenium import webdriver
import urllib
import time

MAX_IMG = 1E4

if __name__ == '__main__':
    driver = webdriver.Chrome()
    n_img = 0   

    for i in range(100):
        url = u'http://stock.tuchong.com/search?term=人物&use=0&type=&layout=&sort=0&page=%d&size=100&search_from=&tp=&abtest=' % i
        driver.get(url)   
        time.sleep(3) 

        row_items = driver.find_elements_by_xpath("//a[@class='row-item']")
        print('got %d row items' % len(row_items))
        for row_item in row_items:
            row_item.click()
            time.sleep(3)
            try:
                img = driver.find_element_by_xpath("//div[@class='image-cover']/img")
                img_src = img.get_attribute('src')
                # download image
                urllib.URLopener().retrieve(img_src, "img/%d.jpg" % n_img)
                n_img += 1
                if n_img > MAX_IMG:
                    break
            except:
                pass
        if n_img >= MAX_IMG:
            break